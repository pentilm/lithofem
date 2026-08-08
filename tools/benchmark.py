"""Benchmark GPU matrix assembly against CPU assembly at three problem sizes.

For each tier of the line-grating example the production solver runs twice with
the cuDSS direct solver: once with `fem.assembly: cpu` (host assembly) and once
with `fem.assembly: gpu` (device assembly + zero-copy handoff). Prints a
markdown table of per-stage timings, the assembly and end-to-end speedups, and
the near-field agreement between the two paths (acceptance criterion: < 1e-10).

Usage: python tools/benchmark.py [outdir]
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lithofem import config, driver, meshgen  # noqa: E402

TIERS = [
    ("small", 1, 8.0),
    ("mid",   2, 10.0),
    ("big",   3, 10.0),
]

GRATING = Path(__file__).resolve().parents[1] / "examples" / "line_grating_te.yaml"
SEGS = ["mesh_read", "rhs", "assemble", "form", "solve", "postproc",
        "output", "total"]


def run(prep, workdir: Path, assembly: str) -> dict:
    env = dict(os.environ)
    per = meshgen.mfem_periodic_mesh_path(prep.mesh_path)
    t0 = time.time()
    res = subprocess.run(
        [str(driver.SOLVER_BIN), "-m", str(per), "-j", str(prep.solve_json_path),
         "-o", str(workdir), "-g", "0", "-d", "cuda", "-a", assembly],
        capture_output=True, text=True, timeout=7200, env=env)
    wall = time.time() - t0
    if res.returncode != 0:
        raise RuntimeError(f"run failed:\n{res.stdout}\n{res.stderr}")
    if assembly == "gpu":
        assert "device-CSR zero-copy path" in res.stdout, "zero-copy not engaged"
        assert "falling back" not in res.stdout, "unexpected fallback"
    meta = json.loads((workdir / "solve_meta_g0.json").read_text())
    t = dict(meta["timing_s"])
    t["wall"] = wall
    t["ndof"] = meta["ndof"]
    t["residual"] = meta["residual"]
    return t


def main() -> None:
    outdir = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/lithofem_bench")
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, order, epw in TIERS:
        with open(GRATING) as f:
            c = yaml.safe_load(f)
        c["fem"] = {"order": order, "elems_per_wavelength": epw}
        c["solver"] = {"type": "direct", "device": "gpu", "gpu_ids": [0]}
        model = config.expand(c)
        base = outdir / f"tier_{name}"
        prep = driver.prepare(model, base)
        cdir, gdir = base / "cpu_asm", base / "gpu_asm"
        cdir.mkdir(exist_ok=True)
        gdir.mkdir(exist_ok=True)
        cpu = run(prep, cdir, "cpu")
        gpu = run(prep, gdir, "gpu")
        field_rel = 0.0
        for p in (0, 1):
            a = np.fromfile(gdir / f"plane_g0_p{p}.bin")
            b = np.fromfile(cdir / f"plane_g0_p{p}.bin")
            field_rel = max(field_rel, float(np.max(np.abs(a - b)) /
                                             np.max(np.abs(b))))
        rows.append((name, cpu, gpu, field_rel))
        print(f"[{name}] ndof={cpu['ndof']} "
              f"assembly {cpu['assemble']:.1f} -> {gpu['assemble']:.1f} s "
              f"(x{cpu['assemble'] / gpu['assemble']:.1f}) "
              f"end-to-end {cpu['total']:.1f} -> {gpu['total']:.1f} s "
              f"field agreement {field_rel:.1e}", flush=True)

    hdr = " | ".join(SEGS)
    print(f"\n| tier | complex DOF | assembly | {hdr} | field vs CPU |")
    print("|---|---|---|" + "---|" * len(SEGS) + "---|")
    for name, cpu, gpu, frel in rows:
        for tag, t in (("cpu", cpu), ("gpu", gpu)):
            cells = " | ".join(f"{t.get(s, 0.0):.1f}" for s in SEGS)
            fr = f"{frel:.1e}" if tag == "gpu" else "-"
            print(f"| {name} | {t['ndof']:,} | {tag} | {cells} | {fr} |")
    (outdir / "bench.json").write_text(json.dumps(
        [{"tier": n, "cpu_asm": c, "gpu_asm": g, "field_rel": f}
         for n, c, g, f in rows], indent=1))


if __name__ == "__main__":
    main()
