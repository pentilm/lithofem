"""V2-M2 acceptance tests: cuDSS GPU direct solve integration.

Criteria (docs/gpu.md, V2-M2):
  - representative regression cases (M6-2 difference layer, M8-1 line
    grating, M8-4 3D square hole): GPU direct vs CPU direct observation
    fields and order efficiencies rel < 1e-8; GPU path self-residual
    < 1e-10; the log must prove cuDSS actually ran (no silent fallback);
  - per_source multi-RHS: one factorization + reuse (log assert), combined
    solution vs per-source sum < 1e-10 (M6b-4 criterion);
  - injected failure -> automatic UMFPACK fallback completes with correct
    result (< 1e-8) and an explicit log message;
  - gpu_ids routing asserted from the cudss device banner.

All tests are marked gpu+cudss and auto-skip when the machine has neither.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest

from lithofem import config, driver, meshgen, outputs

from . import test_solver_m6 as m6
from . import test_solver_m6b as m6b
from . import test_solver_m8 as m8

pytestmark = [
    pytest.mark.full,
    pytest.mark.gpu,
    pytest.mark.cudss,
    pytest.mark.skipif(not driver.SOLVER_BIN.exists(),
                       reason="lithofem_solve not built"),
]

GPU_SOLVER = {"type": "direct", "device": "gpu", "gpu_ids": [0]}
CPU_SOLVER = {"type": "direct", "device": "cpu"}


def _run(c: dict, tmp: Path, extra_env: dict | None = None,
         extra: list[str] | None = None):
    """prepare + raw solver run; returns (stdout, prep)."""
    model = config.expand(c)
    prep = driver.prepare(model, tmp)
    per = meshgen.mfem_periodic_mesh_path(prep.mesh_path)
    env = dict(os.environ)
    env.update(extra_env or {})
    res = subprocess.run(
        [str(driver.SOLVER_BIN), "-m", str(per), "-j",
         str(prep.solve_json_path), "-o", str(prep.workdir), "-g", "0",
         *(extra or [])],
        capture_output=True, text=True, timeout=3600, env=env)
    assert res.returncode == 0, res.stdout + res.stderr
    return res.stdout, prep


def _residual(stdout: str) -> float:
    return float(re.search(r"^residual ([\d.eE+-]+)", stdout, re.M).group(1))


def _assert_cudss_engaged(stdout: str) -> None:
    assert "cudss_factor_s" in stdout, "cuDSS did not run"
    assert "falling back to UMFPACK" not in stdout, "silent CPU fallback"
    assert "cudss_device 0" in stdout  # gpu_ids[0] routing


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def _plane_rels(prep_gpu, prep_cpu, nplanes: int) -> list[float]:
    return [
        _rel(driver.load_plane_envelope(prep_gpu, 0, p),
             driver.load_plane_envelope(prep_cpu, 0, p))
        for p in range(nplanes)
    ]


def test_v2m2_diff_layer_gpu_vs_cpu(tmp_path: Path) -> None:
    """M6-2 difference layer (multilayer with TMM-validated physics)."""
    c = m6._multilayer_config(13.5, 6.0, "s", diff_layer=True)
    c["fem"] = {"order": 3, "elems_per_wavelength": 6}
    out_gpu, prep_gpu = _run({**c, "solver": GPU_SOLVER}, tmp_path / "gpu")
    out_cpu, prep_cpu = _run({**c, "solver": CPU_SOLVER}, tmp_path / "cpu")
    _assert_cudss_engaged(out_gpu)
    assert _residual(out_gpu) < 1e-10, out_gpu
    rels = _plane_rels(prep_gpu, prep_cpu, 1)
    assert max(rels) < 1e-8, rels


def test_v2m2_line_grating_gpu_vs_cpu(tmp_path: Path) -> None:
    """M8-1 line grating: fields AND diffraction order efficiencies."""
    c = m8._grating_config(0.5, "TE", order=2, epw=8.0)
    out_gpu, prep_gpu = _run({**c, "solver": GPU_SOLVER}, tmp_path / "gpu")
    out_cpu, prep_cpu = _run({**c, "solver": CPU_SOLVER}, tmp_path / "cpu")
    _assert_cudss_engaged(out_gpu)
    assert _residual(out_gpu) < 1e-10, out_gpu
    rels = _plane_rels(prep_gpu, prep_cpu, 2)
    assert max(rels) < 1e-8, rels
    for plane, sgn in ((0, +1), (1, -1)):
        eff_gpu = outputs.order_efficiencies(prep_gpu, 0, plane, 1.0 + 0j, sgn)
        eff_cpu = outputs.order_efficiencies(prep_cpu, 0, plane, 1.0 + 0j, sgn)
        assert set(eff_gpu) == set(eff_cpu)
        for k in eff_cpu:
            # 1e-8 relative for physically significant orders; absolute for the
            # evanescent tail, where efficiencies are numerical zeros (~1e-12)
            # and a relative criterion would demand agreement below double
            # precision (same convention as test_solver_m8._compare)
            assert abs(eff_gpu[k] - eff_cpu[k]) <= 1e-8 * max(abs(eff_cpu[k]), 1e-3), (
                k, eff_gpu[k], eff_cpu[k])


def test_v2m2_square_hole_gpu_vs_cpu(tmp_path: Path) -> None:
    """M8-4 3D square hole (true 3D pattern, sloped walls)."""
    c = m8._hole_config(m8.SQUARE, 2)
    out_gpu, prep_gpu = _run({**c, "solver": GPU_SOLVER}, tmp_path / "gpu")
    out_cpu, prep_cpu = _run({**c, "solver": CPU_SOLVER}, tmp_path / "cpu")
    _assert_cudss_engaged(out_gpu)
    assert _residual(out_gpu) < 1e-10, out_gpu
    rels = _plane_rels(prep_gpu, prep_cpu, 2)
    assert max(rels) < 1e-8, rels
    eff_gpu = outputs.order_efficiencies(prep_gpu, 0, 0, 1.0 + 0j, +1)
    eff_cpu = outputs.order_efficiencies(prep_cpu, 0, 0, 1.0 + 0j, +1)
    for k in eff_cpu:
        assert abs(eff_gpu[k] - eff_cpu[k]) <= 1e-8 * max(abs(eff_cpu[k]), 1e-3)


def test_v2m2_per_source_factor_reuse(tmp_path: Path) -> None:
    """M6b-4 superposition on the GPU: factorize once, reuse for each RHS."""
    c = m6b._sheet_config()
    c["sources"].append({
        "type": "sheet", "corner": [0.0, 0.0, 24.0],
        "edges": [[m6b.LX, 0.0, 0.0], [0.0, m6b.LX, 0.0]],
        "current": [[0, 0], [1, 0.5], [0, 0]],
    })
    c["output"]["per_source"] = True
    c["solver"] = GPU_SOLVER
    out, prep = _run(c, tmp_path)
    _assert_cudss_engaged(out)
    # exactly one factorization; every additional RHS reuses it
    assert out.count("cudss_factor_s") == 1, out
    assert out.count("cudss: factorization reused") >= 2, out
    for plane in (0, 1):
        combined = driver.load_plane_envelope(prep, 0, plane)
        total = np.zeros_like(combined)
        for si in (0, 1):
            raw = np.fromfile(
                prep.workdir / f"plane_g0_p{plane}_s{si}.bin", dtype=np.float64
            ).reshape(*combined.shape, 2)
            total += raw[..., 0] + 1j * raw[..., 1]
        scale = np.abs(combined).max()
        assert np.abs(combined - total).max() < 1e-10 * max(scale, 1.0)


def test_v2m2_forced_failure_falls_back_to_umfpack(tmp_path: Path) -> None:
    c = m6._multilayer_config(13.5, 6.0, "s", diff_layer=True)
    c["fem"] = {"order": 2, "elems_per_wavelength": 5}
    out_cpu, prep_cpu = _run({**c, "solver": CPU_SOLVER}, tmp_path / "cpu")
    out_fb, prep_fb = _run({**c, "solver": GPU_SOLVER}, tmp_path / "fb",
                           extra_env={"LITHOFEM_CUDSS_FORCE_FAIL": "1"})
    assert "cudss: forced failure injected" in out_fb
    assert "falling back to UMFPACK" in out_fb
    assert "cudss_factor_s" not in out_fb
    assert _residual(out_fb) < 1e-10
    rels = _plane_rels(prep_fb, prep_cpu, 1)
    assert max(rels) < 1e-8, rels


def test_v2m2_gpu_ids_banner(tmp_path: Path) -> None:
    """gpu_ids[0] reaches cudaSetDevice (single-GPU machine: id 0)."""
    c = m6b._sheet_config()
    c["fem"] = {"order": 2, "elems_per_wavelength": 6}
    c["solver"] = GPU_SOLVER
    out, _ = _run(c, tmp_path)
    _assert_cudss_engaged(out)
    assert re.search(r"^cudss_device 0 ", out, re.M), out
