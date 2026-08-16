"""V2-M4 acceptance tests: task-level sweep scheduler (driver.sweep_solve).

Criteria (docs/gpu.md, V2-M4):
  - >=3 k-parallel groups, gpu_ids [0, 0] (single card emulating two slots),
    process-level concurrency vs serial: identical outputs < 1e-12, archive
    naming/logs intact;
  - failure isolation: one injected failing task -> the others complete and
    the aggregate error localizes the failure;
  - max_parallel cap enforced (asserted from task start/end intervals);
  - every task log records its CUDA_VISIBLE_DEVICES binding (single-GPU
    machine: physical id 0; multi-GPU assertions are the ⊕ item in TRACE-V2).

The scheduler is device-agnostic; scheduling tests run on CPU so they cover
non-GPU machines too. The GPU binding test is marked gpu+cudss.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from lithofem import config, driver

pytestmark = [
    pytest.mark.full,
    pytest.mark.skipif(not driver.SOLVER_BIN.exists(),
                       reason="lithofem_solve not built"),
]


def _multi_group_config(solver: dict | None = None) -> dict:
    c = {
        "domain": {"Lx": 10.0, "Ly": 10.0, "z_min": 0.0, "z_max": 40.0},
        "materials": {"m": {"epsilon": [2.0, 0.4]}},
        "layers": [{"z": [0.0, 15.0], "material": "m"}],
        "frustums": [],
        "wavelength": 50.0,
        "sources": [
            {"type": "planewave",
             "incidence": {"theta": th, "phi": 0, "from": "top"},
             "polarization": "s"}
            for th in (0.0, 6.0, 12.0)   # three distinct kpar groups
        ],
        "output": {"planes": [{"z": 32.0, "quantities": ["E"],
                               "resolution": [8, 8], "file": "o.h5"}]},
        "fem": {"order": 2, "elems_per_wavelength": 4},
        "sweep": {"gpu_ids": [0, 0], "max_parallel": 2},
    }
    if solver:
        c["solver"] = solver
    return c


def _max_overlap(results: list[driver.SweepTask]) -> int:
    """Maximum number of simultaneously running tasks."""
    events = [(r.t_start, +1) for r in results] + [(r.t_end, -1) for r in results]
    peak = cur = 0
    for _, d in sorted(events):
        cur += d
        peak = max(peak, cur)
    return peak


def _snapshot_planes(prep, groups) -> dict[int, np.ndarray]:
    return {g: driver.load_plane_envelope(prep, g, 0).copy() for g in groups}


def test_v2m4_concurrent_vs_serial(tmp_path: Path) -> None:
    model = config.expand(_multi_group_config())
    assert len(model.groups) == 3
    prep = driver.prepare(model, tmp_path)

    results = driver.sweep_solve(prep)   # sweep from model: [0,0] x 2
    assert all(r.ok for r in results), driver.sweep_summary_error(results)
    assert [r.group for r in results] == [0, 1, 2]
    assert _max_overlap(results) <= 2
    swept = _snapshot_planes(prep, range(3))
    # archive contract: per-group logs + meta written
    for g in range(3):
        log = (prep.workdir / f"sweep_g{g}.log").read_text()
        assert "CUDA_VISIBLE_DEVICES=0" in log
        assert f"group={g}" in log
        assert results[g].meta is not None and "residual" in results[g].meta

    # serial reference on the SAME mesh/workdir (files are overwritten)
    for g in range(3):
        driver.solve_group(prep, g)
    for g in range(3):
        serial = driver.load_plane_envelope(prep, g, 0)
        scale = np.abs(serial).max()
        diff = np.abs(swept[g] - serial).max()
        assert diff < 1e-12 * max(scale, 1.0), (g, diff, scale)


def test_v2m4_failure_isolation(tmp_path: Path) -> None:
    model = config.expand(_multi_group_config())
    prep = driver.prepare(model, tmp_path)
    results = driver.sweep_solve(prep, groups=[0, 1, 99])  # 99: no such group
    assert results[0].ok and results[1].ok
    assert not results[2].ok
    err = driver.sweep_summary_error(results)
    assert "1/3" in err and "group 99" in err and "sweep_g99.log" in err


def test_v2m4_max_parallel_cap(tmp_path: Path) -> None:
    c = _multi_group_config()
    c["sweep"] = {"gpu_ids": [0], "max_parallel": 1}
    model = config.expand(c)
    prep = driver.prepare(model, tmp_path)
    results = driver.sweep_solve(prep)
    assert all(r.ok for r in results)
    assert _max_overlap(results) == 1   # strictly serialized


@pytest.mark.gpu
@pytest.mark.cudss
@pytest.mark.multigpu
def test_v2m4_r2_10_multi_gpu_physical(tmp_path: Path) -> None:
    """R2-10 (V2-M4 ⊕): real multi-GPU sweep — distinct physical card
    bindings, results identical to serial, and a concurrency speedup.
    Runs through the v2.5 GPU-assembly + zero-copy path on every card.
    Auto-skips on machines with fewer than two GPUs."""
    import time

    from . import conftest
    if conftest.GPU_COUNT < 2:
        pytest.skip("needs >= 2 physical GPUs")

    c = _multi_group_config({"type": "direct", "device": "gpu",
                             "gpu_ids": [0]})
    c["fem"] = {"order": 2, "elems_per_wavelength": 6, "assembly": "gpu"}
    c["sweep"] = {"gpu_ids": [0, 1], "max_parallel": 2}
    model = config.expand(c)
    prep = driver.prepare(model, tmp_path)

    t0 = time.time()
    results = driver.sweep_solve(prep)
    t_par = time.time() - t0
    assert all(r.ok for r in results), driver.sweep_summary_error(results)
    assert _max_overlap(results) == 2

    # distinct physical cards actually used (round-robin 0,1,0) and the
    # full v2.5 pipeline engaged inside every pinned task
    cvds = set()
    for g in range(3):
        log = (prep.workdir / f"sweep_g{g}.log").read_text()
        m = re.search(r"CUDA_VISIBLE_DEVICES=(\d+)", log)
        cvds.add(m.group(1))
        assert "cudss_device 0" in log          # ordinal inside the pin
        assert "device-CSR zero-copy path" in log
        assert "falling back" not in log
    assert cvds == {"0", "1"}
    assert results[0].gpu_id == 0 and results[1].gpu_id == 1
    assert results[2].gpu_id == 0

    swept = _snapshot_planes(prep, range(3))

    # serial reference on card 0 (same mesh/workdir) + throughput baseline
    t0 = time.time()
    for g in range(3):
        driver.solve_group(prep, g)
    t_serial = time.time() - t0
    for g in range(3):
        serial = driver.load_plane_envelope(prep, g, 0)
        scale = np.abs(serial).max()
        diff = np.abs(swept[g] - serial).max()
        assert diff < 1e-12 * max(scale, 1.0), (g, diff, scale)
    # two cards, three tasks: wall time must beat strict serial
    print(f"r2_10_t_par {t_par:.2f} r2_10_t_serial {t_serial:.2f} "
          f"speedup {t_serial / t_par:.2f}")
    assert t_par < t_serial, (t_par, t_serial)


@pytest.mark.gpu
@pytest.mark.cudss
def test_v2m4_gpu_binding(tmp_path: Path) -> None:
    """Each task binds CUDA_VISIBLE_DEVICES and cuDSS reports device 0
    (the only visible device inside the task)."""
    model = config.expand(
        _multi_group_config({"type": "direct", "device": "gpu", "gpu_ids": [0]}))
    prep = driver.prepare(model, tmp_path)
    results = driver.sweep_solve(prep)
    assert all(r.ok for r in results), driver.sweep_summary_error(results)
    for g in range(3):
        log = (prep.workdir / f"sweep_g{g}.log").read_text()
        assert "CUDA_VISIBLE_DEVICES=0" in log
        assert "cudss_device 0" in log
        assert "cudss_factor_s" in log
        assert "falling back" not in log
