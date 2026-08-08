"""Shared test configuration: auto-skip GPU tests when no CUDA device."""

from __future__ import annotations

import shutil
import subprocess

import pytest


def _gpu_count() -> int:
    if not shutil.which("nvidia-smi"):
        return 0
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True,
                             text=True, timeout=20)
        if out.returncode != 0:
            return 0
        return sum(1 for ln in out.stdout.splitlines() if ln.startswith("GPU "))
    except (OSError, subprocess.TimeoutExpired):
        return 0


GPU_COUNT = _gpu_count()
HAS_GPU = GPU_COUNT > 0


def _has_cudss() -> bool:
    """cuDSS runtime present (pip wheel `nvidia-cudss-cu13`, design note)?"""
    if not HAS_GPU:
        return False
    import ctypes
    import glob

    candidates = glob.glob(
        "/opt/conda/lib/python3*/site-packages/nvidia/cu13/lib/libcudss.so*"
    ) + glob.glob("/usr/local/cuda*/lib64/libcudss.so*")
    for path in candidates:
        try:
            ctypes.CDLL(path)
            return True
        except OSError:
            continue
    return False


HAS_CUDSS = _has_cudss()


def pytest_collection_modifyitems(config, items):  # noqa: ANN001
    skip_gpu = pytest.mark.skip(reason="no CUDA GPU detected (auto-skip)")
    skip_cudss = pytest.mark.skip(reason="cuDSS not available (auto-skip)")
    for item in items:
        if not HAS_GPU and "gpu" in item.keywords:
            item.add_marker(skip_gpu)
        if not HAS_CUDSS and "cudss" in item.keywords:
            item.add_marker(skip_cudss)
