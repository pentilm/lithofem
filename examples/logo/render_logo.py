"""Render the simulated near fields into logo images.

Reads the HDF5 observation planes written by the run and produces:
  logo_<plane>.png   - intensity |E|^2 for each observation distance
  logo_strip.png     - the three distances stacked, showing diffraction growth
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/logo/out")
DEST = Path(sys.argv[2] if len(sys.argv) > 2 else "/workspace/logo")

# deep blue -> cyan -> warm white: reads as "light coming through"
CMAP = LinearSegmentedColormap.from_list("litho", [
    (0.00, "#05070f"),
    (0.25, "#0d2a4a"),
    (0.50, "#1f7fb0"),
    (0.72, "#57c8d8"),
    (0.88, "#bdeef0"),
    (1.00, "#ffffff"),
])


def load_intensity(path: Path) -> np.ndarray:
    """|E|^2 on the plane, shape (ny, nx)."""
    with h5py.File(path, "r") as f:
        e = np.asarray(f["E_re"]) + 1j * np.asarray(f["E_im"])   # (ny, nx, 3)
    return np.sum(np.abs(e) ** 2, axis=-1)


def tile(img: np.ndarray, reps: int = 2) -> np.ndarray:
    """Repeat laterally: the cell is periodic, so the mask really does tile."""
    return np.tile(img, (1, reps))


def norm(img: np.ndarray, lo: float = 1.0, hi: float = 99.6) -> np.ndarray:
    a, b = np.percentile(img, lo), np.percentile(img, hi)
    return np.clip((img - a) / max(b - a, 1e-30), 0.0, 1.0)


def save(img: np.ndarray, path: Path, height_in: float = 2.6) -> None:
    ny, nx = img.shape
    fig = plt.figure(figsize=(height_in * nx / ny, height_in), dpi=180)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.imshow(norm(img), cmap=CMAP, origin="lower", interpolation="bilinear",
              aspect="auto")
    ax.axis("off")
    fig.savefig(path, facecolor="#05070f")
    plt.close(fig)
    print(f"  {path.name}  ({nx} x {ny})")


def main() -> None:
    names = ["near", "mid", "far"]
    imgs = {}
    for n in names:
        cands = sorted(OUT.glob(f"*logo_{n}*.h5"))
        if not cands:
            print(f"missing plane: {n}")
            continue
        imgs[n] = tile(load_intensity(cands[0]))
        save(imgs[n], DEST / f"logo_{n}.png")

    if len(imgs) == len(names):
        gap = np.zeros((max(8, imgs["near"].shape[0] // 12),
                        imgs["near"].shape[1]))
        stack = []
        for i, n in enumerate(names):
            stack.append(norm(imgs[n]))
            if i < len(names) - 1:
                stack.append(gap)
        save(np.vstack(stack), DEST / "logo_strip.png", height_in=6.0)


if __name__ == "__main__":
    main()
