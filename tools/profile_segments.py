"""Per-stage timing profile of the solver at three problem sizes.

Runs the line-grating example at three resolutions with both the CPU and the
GPU direct solver, reads the `timing_s` block written into the solve metadata
(mesh read / FE space / RHS / assembly / system formation / solve /
post-processing / output), and prints a markdown table. Useful for finding which
stage dominates before optimizing anything.

Usage: python tools/profile_segments.py [outdir]
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lithofem import config, driver, meshgen  # noqa: E402

TIERS = [
    ("small", 1, 8.0),
    ("mid",   2, 10.0),
    ("big",   3, 10.0),
]

GRATING = Path(__file__).resolve().parents[1] / "examples" / "line_grating_te.yaml"

SEGS = ["mesh_read", "fespace", "rhs", "assemble", "form", "solve",
        "postproc", "output", "total"]


def run(prep, device: str, workdir: Path, threads_cap: int | None) -> dict:
    per = meshgen.mfem_periodic_mesh_path(prep.mesh_path)
    env = dict(os.environ)
    if threads_cap:
        # Large CPU factorizations are unstable with very high OpenBLAS thread
        # counts; cap them for the CPU reference runs.
        env["OPENBLAS_NUM_THREADS"] = str(threads_cap)
    t0 = time.time()
    res = subprocess.run(
        [str(driver.SOLVER_BIN), "-m", str(per), "-j", str(prep.solve_json_path),
         "-o", str(workdir), "-g", "0", "-d", device],
        capture_output=True, text=True, timeout=7200, env=env)
    wall = time.time() - t0
    if res.returncode != 0:
        raise RuntimeError(f"{device} run failed:\n{res.stdout}\n{res.stderr}")
    meta = json.loads((workdir / "solve_meta_g0.json").read_text())
    t = dict(meta["timing_s"])
    t["wall"] = wall
    t["ndof"] = meta["ndof"]
    return t


def main() -> None:
    outdir = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/lithofem_profile")
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, order, epw in TIERS:
        with open(GRATING) as f:
            c = yaml.safe_load(f)
        c["fem"] = {"order": order, "elems_per_wavelength": epw}
        model = config.expand(c)
        wdir = outdir / f"tier_{name}"
        prep = driver.prepare(model, wdir)
        cap = 32 if order >= 3 else None
        cpu = run(prep, "cpu", wdir, cap)
        gpu = run(prep, "cuda", wdir, None)
        rows.append((name, cpu, gpu))
        print(f"[{name}] ndof={cpu['ndof']} cpu_total={cpu['total']:.1f}s "
              f"(assembly {cpu['assemble']:.1f}s) "
              f"gpu_total={gpu['total']:.1f}s (assembly {gpu['assemble']:.1f}s)",
              flush=True)

    hdr = " | ".join(SEGS)
    print(f"\n| tier | complex DOF | solver | {hdr} |")
    print("|---|---|---|" + "---|" * len(SEGS))
    for name, cpu, gpu in rows:
        for dev, t in (("cpu", cpu), ("gpu", gpu)):
            cells = " | ".join(f"{t.get(s, 0.0):.1f}" for s in SEGS)
            print(f"| {name} | {t['ndof']:,} | {dev} | {cells} |")
    (outdir / "profile.json").write_text(json.dumps(
        [{"tier": n, "cpu": c, "gpu": g} for n, c, g in rows], indent=1))


if __name__ == "__main__":
    main()
