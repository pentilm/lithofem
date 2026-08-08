"""CUDA availability probe (M0). Prints a machine-readable summary.

Checked layers: driver (nvidia-smi), toolkit (nvcc), MFEM build flags.
Exit code 0 always — absence of a GPU is a valid, reported state.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

MFEM_CONFIG = Path("/workspace/mfem-4.9/config/config.mk")
NVCC_CANDIDATES = ["nvcc", "/usr/local/cuda/bin/nvcc"]


def run(cmd: list[str]) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None


def main() -> None:
    query = ["--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]
    smi = run(["nvidia-smi", *query])
    gpus = [line.strip() for line in smi.splitlines() if line.strip()] if smi else []
    print(f"driver_gpus: {len(gpus)}")
    for i, g in enumerate(gpus):
        print(f"  gpu[{i}]: {g}")

    nvcc = next((c for c in NVCC_CANDIDATES if shutil.which(c) or Path(c).exists()), None)
    ver = run([nvcc, "--version"]) if nvcc else None
    m = re.search(r"release ([\d.]+)", ver or "")
    print(f"nvcc: {nvcc or 'NOT FOUND'} (release {m.group(1) if m else '?'})")

    mfem_cuda = None
    if MFEM_CONFIG.exists():
        text = MFEM_CONFIG.read_text()
        mfem_cuda = bool(re.search(r"^MFEM_USE_CUDA\s*=\s*YES", text, re.M))
        arch = re.search(r"-arch=(\S+)", text)
        print(f"mfem_use_cuda: {mfem_cuda} (arch {arch.group(1) if arch else '?'})")
    else:
        print("mfem_use_cuda: config.mk not found")

    usable = bool(gpus) and nvcc is not None and bool(mfem_cuda)
    print(f"verdict: {'CUDA-usable' if usable else 'CPU-only'}")


if __name__ == "__main__":
    main()
