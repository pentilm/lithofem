"""Generate a LithoFEM configuration whose mask cut-outs spell "LithoFEM".

Letters are built on a stroke-unit grid as sets of axis-aligned rectangles
(plus one outline polygon for M). LithoFEM's frustums may not overlap, but
edge-touching is fine (the check is on intersection *area*), so a letter can be
assembled from adjacent bars. Enclosed counters are not supported (no holes),
so 'o' is a square ring made of four bars.

The letters are vacuum cut-outs in an absorber film; illuminating from the top
and observing below gives the diffracted "aerial image" of the word.
"""

from __future__ import annotations

import sys

import yaml

WL = 193.0                 # DUV wavelength, nm (physics is scale invariant)
U = 0.60 * WL              # stroke unit = 0.6 lambda
GAP = 0.9                  # inter-letter gap, in stroke units
MARGIN_X, MARGIN_Y = 1.2, 1.1   # cell margins, in stroke units

# Each letter: list of rectangles (x0, y0, x1, y1) in stroke units, plus its
# advance width. Cap height 5, x-height 3. Rectangles never overlap.
LETTERS: dict[str, tuple[float, list]] = {
    "L": (3.0, [(0, 0, 1, 5), (1, 0, 3, 1)]),
    "i": (1.0, [(0, 0, 1, 3), (0, 4, 1, 5)]),
    "t": (3.0, [(1, 0, 2, 5), (0, 3, 1, 4), (2, 3, 3, 4)]),
    "h": (3.0, [(0, 0, 1, 5), (1, 2, 2, 3), (2, 0, 3, 3)]),
    # square 'o': a box ring, four non-overlapping bars
    "o": (3.0, [(0, 0, 3, 1), (0, 2, 3, 3), (0, 1, 1, 2), (2, 1, 3, 2)]),
    "F": (3.0, [(0, 0, 1, 5), (1, 4, 3, 5), (1, 2, 2.6, 3)]),
    "E": (3.0, [(0, 0, 1, 5), (1, 4, 3, 5), (1, 2, 2.6, 3), (1, 0, 3, 1)]),
}

# M as a single simple outline polygon (vertices in stroke units)
M_ADVANCE = 4.0
M_OUTLINE = [
    (0, 0), (1, 0), (1, 3.4), (2, 1.3), (3, 3.4), (3, 0), (4, 0),
    (4, 5), (3.05, 5), (2, 2.7), (0.95, 5), (0, 5),
]

WORD = "LithoFEM"


def layout() -> tuple[list[list[tuple[float, float]]], float, float]:
    """Return polygons (in nm, origin at 0) plus the text bounding box."""
    polys: list[list[tuple[float, float]]] = []
    x = 0.0
    for ch in WORD:
        if ch == "M":
            polys.append([(x + px, py) for px, py in M_OUTLINE])
            x += M_ADVANCE + GAP
            continue
        adv, rects = LETTERS[ch]
        for x0, y0, x1, y1 in rects:
            polys.append([(x + x0, y0), (x + x1, y0), (x + x1, y1), (x + x0, y1)])
        x += adv + GAP
    text_w = x - GAP
    return polys, text_w, 5.0


def build_config() -> dict:
    polys_u, text_w_u, text_h_u = layout()

    lx = (text_w_u + 2 * MARGIN_X) * U
    ly = (text_h_u + 2 * MARGIN_Y) * U
    ox, oy = MARGIN_X * U, MARGIN_Y * U

    mask_t = 0.42 * WL          # absorber thickness
    air_below = 1.0 * WL        # propagation space for the diffracted image
    air_above = 0.55 * WL

    frustums = []
    for p in polys_u:
        verts = [[round(ox + px * U, 4), round(oy + py * U, 4)] for px, py in p]
        frustums.append({
            "vertices": verts,
            "z0": 0.0,
            "h": mask_t,
            "alpha": 90,          # vertical walls: no mitre-offset constraint
            "epsilon": 1.0,       # vacuum cut-out -> this is where light passes
        })

    # observation planes: right under the mask, then progressively farther,
    # so the image goes from sharp to visibly diffracted
    planes = []
    for frac, name in ((0.12, "near"), (0.45, "mid"), (0.8, "far")):
        planes.append({
            "z": round(-frac * WL, 3),
            "quantities": ["E"],
            "resolution": [900, 220],
            "file": f"logo_{name}.h5",
        })

    return {
        "schema_version": 1,
        "domain": {"Lx": round(lx, 3), "Ly": round(ly, 3),
                   "z_min": round(-air_below, 3), "z_max": round(mask_t + air_above, 3)},
        "materials": {
            # chrome-like DUV absorber: strong amplitude contrast
            "absorber": {"n": 1.48, "k": 1.79},
        },
        "layers": [{"z": [0.0, round(mask_t, 3)], "material": "absorber"}],
        "frustums": frustums,
        "wavelength": WL,
        "sources": [{
            "type": "planewave",
            "incidence": {"theta": 0, "phi": 0, "from": "top"},
            "polarization": "s",
        }],
        "fem": {"order": 2, "elems_per_wavelength": float(sys.argv[2])
                if len(sys.argv) > 2 else 4.0, "assembly": "gpu"},
        "solver": {"type": "direct", "device": "gpu", "gpu_ids": [0]},
        "boundaries": {"pml": {"thickness": 0.6, "target_reflection": 1e-8}},
        "output": {"planes": planes, "orders": {"enabled": False}},
    }


if __name__ == "__main__":
    cfg = build_config()
    out = sys.argv[1] if len(sys.argv) > 1 else "/workspace/logo/logo.yaml"
    with open(out, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, width=200)
    d = cfg["domain"]
    print(f"wrote {out}")
    print(f"  cell {d['Lx']:.0f} x {d['Ly']:.0f} nm "
          f"= {d['Lx']/WL:.1f} x {d['Ly']/WL:.1f} lambda")
    print(f"  z {d['z_min']:.0f} .. {d['z_max']:.0f} nm, "
          f"{len(cfg['frustums'])} cut-out polygons")
