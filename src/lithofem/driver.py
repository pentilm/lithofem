"""Pipeline driver: config -> mesh + solve.json -> C++ solve -> results.

Stage boundaries follow docs/physics.md The solve.json written here extends
config.model_to_solve_json with the per-group incident tables (M6).
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import config as cfg
from . import incident, meshgen

SOLVER_BIN = Path(__file__).resolve().parents[2] / "solver" / "bin" / "lithofem_solve"


def full_solve_json(model: cfg.Model) -> dict[str, Any]:
    doc = cfg.model_to_solve_json(model)
    for gi, group in enumerate(model.groups):
        inc = incident.group_incident(model, group)
        doc["groups"][gi]["incident"] = incident.incident_to_json(inc)
        doc["groups"][gi]["r_amp"] = [inc.r_amp.real, inc.r_amp.imag]
        doc["groups"][gi]["t_amp"] = [inc.t_amp.real, inc.t_amp.imag]
    return doc


@dataclass
class Prepared:
    model: cfg.Model
    workdir: Path
    mesh_path: Path
    solve_json_path: Path
    mesh_info: meshgen.MeshInfo


def prepare(model: cfg.Model, workdir: str | Path) -> Prepared:
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    mesh_path = workdir / "mesh.msh"
    info = meshgen.generate(model, mesh_path)
    sj = workdir / "solve.json"
    with open(sj, "w") as f:
        json.dump(full_solve_json(model), f, indent=1)
    return Prepared(model=model, workdir=workdir, mesh_path=mesh_path,
                    solve_json_path=sj, mesh_info=info)


def solve_group(
    prep: Prepared, group: int = 0, device: str = "",
    solver_bin: Path | None = None, timeout: int = 3600,
    shift_x: float = 0.0, assembly: str = "",
) -> dict[str, Any]:
    """Run the C++ solver for one group; returns its meta dict."""
    binp = solver_bin or SOLVER_BIN
    per_mesh = meshgen.mfem_periodic_mesh_path(prep.mesh_path)
    res = subprocess.run(
        [str(binp), "-m", str(per_mesh), "-j", str(prep.solve_json_path),
         "-o", str(prep.workdir), "-g", str(group), "-d", device,
         "-sx", str(shift_x)]
        + (["-a", assembly] if assembly else []),
        capture_output=True, text=True, timeout=timeout,
    )
    if res.returncode != 0:
        raise RuntimeError(
            f"lithofem_solve failed (rc={res.returncode}):\n{res.stdout}\n{res.stderr}"
        )
    with open(prep.workdir / f"solve_meta_g{group}.json") as f:
        meta = json.load(f)
    meta["stdout"] = res.stdout
    return meta


@dataclass
class SweepTask:
    """Outcome of one sweep task (V2-M4). Failures are isolated: `ok` is
    False and `error` holds the diagnostic; other tasks are unaffected."""
    group: int
    gpu_id: int
    ok: bool
    seconds: float
    t_start: float
    t_end: float
    error: str = ""
    meta: dict[str, Any] | None = None


def sweep_solve(
    prep: Prepared, sweep: cfg.SweepParams | None = None,
    groups: list[int] | None = None, device: str = "",
    solver_bin: Path | None = None, timeout: int = 3600,
) -> list[SweepTask]:
    """Task-level sweep (docs/gpu.md): one solver subprocess per group,
    round-robin over sweep.gpu_ids, at most sweep.max_parallel concurrent.

    Each task runs with CUDA_VISIBLE_DEVICES bound to its GPU and logs to
    workdir/sweep_g<g>.log (binding + full solver output). Failures are
    isolated; inspect the returned list (order matches `groups`).
    """
    sweep = sweep or prep.model.sweep or cfg.SweepParams()
    if groups is None:
        groups = list(range(len(prep.model.groups)))
    binp = solver_bin or SOLVER_BIN
    per_mesh = meshgen.mfem_periodic_mesh_path(prep.mesh_path)

    def run_one(idx: int, g: int) -> SweepTask:
        gid = sweep.gpu_ids[idx % len(sweep.gpu_ids)]
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gid)
        args = [str(binp), "-m", str(per_mesh), "-j", str(prep.solve_json_path),
                "-o", str(prep.workdir), "-g", str(g), "-d", device]
        t0 = time.time()
        try:
            res = subprocess.run(args, capture_output=True, text=True,
                                 timeout=timeout, env=env)
            ok = res.returncode == 0
            err = "" if ok else f"rc={res.returncode}"
            out = res.stdout + res.stderr
        except (OSError, subprocess.TimeoutExpired) as e:
            ok, err, out = False, str(e), ""
        t1 = time.time()
        log_path = prep.workdir / f"sweep_g{g}.log"
        log_path.write_text(
            f"group={g} gpu_id={gid} CUDA_VISIBLE_DEVICES={gid}\n"
            f"cmd={' '.join(args)}\nok={ok} error={err} "
            f"seconds={t1 - t0:.3f}\n---\n{out}")
        meta = None
        if ok:
            try:
                with open(prep.workdir / f"solve_meta_g{g}.json") as f:
                    meta = json.load(f)
            except OSError as e:
                ok, err = False, f"missing solve_meta: {e}"
        return SweepTask(group=g, gpu_id=gid, ok=ok, seconds=t1 - t0,
                         t_start=t0, t_end=t1, error=err, meta=meta)

    # threads only babysit solver subprocesses; the parallelism is process-
    # level (one lithofem_solve per task), capped at max_parallel
    with ThreadPoolExecutor(max_workers=sweep.max_parallel) as ex:
        results = list(ex.map(run_one, range(len(groups)), groups))
    return results


def sweep_summary_error(results: list[SweepTask]) -> str:
    """Aggregate failure message ('' if all tasks succeeded)."""
    failed = [r for r in results if not r.ok]
    if not failed:
        return ""
    lines = [f"sweep: {len(failed)}/{len(results)} task(s) failed:"]
    lines += [f"  group {r.group} (gpu {r.gpu_id}): {r.error} "
              f"[see sweep_g{r.group}.log]" for r in failed]
    return "\n".join(lines)


def load_plane_envelope(prep: Prepared, group: int, plane: int) -> np.ndarray:
    """Scattered-field envelope u on the sampled plane -> (ny, nx, 3) complex."""
    p = prep.model.output.planes[plane]
    nx, ny = p.resolution
    raw = np.fromfile(prep.workdir / f"plane_g{group}_p{plane}.bin", dtype=np.float64)
    raw = raw.reshape(ny, nx, 3, 2)
    return raw[..., 0] + 1j * raw[..., 1]


def plane_grid(model: cfg.Model, plane: int) -> tuple[np.ndarray, np.ndarray]:
    """Cell-centred sample coordinates used by the C++ sampler."""
    p = model.output.planes[plane]
    nx, ny = p.resolution
    x = (np.arange(nx) + 0.5) * model.domain.lx / nx
    y = (np.arange(ny) + 0.5) * model.domain.ly / ny
    return x, y


def total_field_on_plane(
    prep: Prepared, group: int, plane: int
) -> tuple[np.ndarray, np.ndarray]:
    """(E_total, E_sc) on the plane grid, (ny, nx, 3) complex, physical fields."""
    model = prep.model
    u = load_plane_envelope(prep, group, plane)
    x, y = plane_grid(model, plane)
    kx, ky = model.groups[group].kpar
    phase = np.exp(1j * (ky * y[:, None] + kx * x[None, :]))
    e_sc = u * phase[..., None]
    inc = incident.group_incident(model, model.groups[group])
    z = model.output.planes[plane].z
    e_inc = inc.eval(np.array([z]))[0]
    e_tot = e_sc + e_inc[None, None, :] * phase[..., None]
    return e_tot, e_sc
