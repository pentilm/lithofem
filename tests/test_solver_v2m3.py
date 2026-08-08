"""V2-M3 acceptance tests: VRAM estimate accuracy and over-limit fallback.

Criteria (docs/gpu.md, V2-M3):
  - cuDSS analysis-phase VRAM estimate vs measured factor footprint < 2x
    (three scales are tabulated in the acceptance report via tools/v2_bench;
    this test pins one representative case in CI);
  - injected over-limit (-gml) -> warning + automatic CPU fallback with a
    correct result (< 1e-8 vs plain CPU);
  - no-GPU / no-cuDSS environments: auto-skip (conftest markers, R2-12).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from lithofem import driver

from . import test_solver_m6 as m6
from .test_solver_v2 import (
    CPU_SOLVER,
    GPU_SOLVER,
    _assert_cudss_engaged,
    _plane_rels,
    _residual,
    _run,
)

pytestmark = [
    pytest.mark.full,
    pytest.mark.gpu,
    pytest.mark.cudss,
    pytest.mark.skipif(not driver.SOLVER_BIN.exists(),
                       reason="lithofem_solve not built"),
]


def _case() -> dict:
    c = m6._multilayer_config(13.5, 6.0, "s", diff_layer=True)
    c["fem"] = {"order": 2, "elems_per_wavelength": 6}
    return c


def test_v2m3_vram_estimate_within_2x(tmp_path: Path) -> None:
    out, _ = _run({**_case(), "solver": GPU_SOLVER}, tmp_path)
    _assert_cudss_engaged(out)
    est = float(re.search(r"cudss_peak_est_gb ([\d.eE+-]+)", out).group(1))
    used = float(re.search(r"cudss_vram_used_gb ([\d.eE+-]+)", out).group(1))
    assert used > 0.0, out
    ratio = est / used
    assert 0.5 < ratio < 2.0, (est, used, ratio)


def test_v2m3_gpu_mem_limit_fallback(tmp_path: Path) -> None:
    c = _case()
    out_cpu, prep_cpu = _run({**c, "solver": CPU_SOLVER}, tmp_path / "cpu")
    out_fb, prep_fb = _run({**c, "solver": GPU_SOLVER}, tmp_path / "fb",
                           extra=["-gml", "0.05"])
    assert "GPU MEMORY LIMIT" in out_fb, out_fb
    assert "falling back" in out_fb
    assert "cudss_factor_s" not in out_fb   # factorization never ran on GPU
    assert _residual(out_fb) < 1e-10
    rels = _plane_rels(prep_fb, prep_cpu, 1)
    assert max(rels) < 1e-8, rels


def test_v2m3_gpu_mem_limit_config_key(tmp_path: Path) -> None:
    """solver.gpu_mem_gb reaches the solver via solve.json (YAML additive)."""
    c = _case()
    c["solver"] = {**GPU_SOLVER, "gpu_mem_gb": 0.05}
    out, _ = _run(c, tmp_path)
    assert "GPU MEMORY LIMIT" in out and "configured limit is 0.05" in out, out
