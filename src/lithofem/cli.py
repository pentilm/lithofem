"""LithoFEM command-line interface (docs/physics.md, M10).

    lithofem run config.yaml -o results/
    lithofem selftest [--full]
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from . import __version__


def _log(fp: Any, event: str, **kw: Any) -> None:
    fp.write(json.dumps({"event": event, **kw}) + "\n")
    fp.flush()


def run(config_path: str | Path, outdir: str | Path,
        device: str | None = None, assembly: str | None = None) -> Path:
    """Full pipeline: validate -> mesh -> solve (all groups) -> outputs.

    Returns the output directory. Used by both the CLI and the thin Python
    API (lithofem.run), which are therefore identical by construction.
    """
    from . import config as cfg
    from . import driver, outputs

    t_start = time.time()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    model = cfg.load_yaml(config_path)

    # reproducibility: config snapshot + version + expanded solve.json
    shutil.copyfile(config_path, outdir / "config_snapshot.yaml")
    (outdir / "version.json").write_text(json.dumps({
        "lithofem": __version__,
        "python": sys.version.split()[0],
    }, indent=1))

    if model.fem.order >= 6:
        print(f"note: fem.order = {model.fem.order} — assembly and direct-"
              "solve cost grow quickly and conditioning degrades at high "
              "order; consider a coarser mesh (docs/configuration.md)")

    with open(outdir / "run_log.jsonl", "w") as log:
        _log(log, "start", config=str(config_path), version=__version__)
        t0 = time.time()
        prep = driver.prepare(model, outdir)
        _log(log, "mesh", seconds=round(time.time() - t0, 3),
             regions=prep.mesh_info.n_regions,
             region_volumes=prep.mesh_info.region_volumes)

        solved_groups = list(range(len(model.groups)))
        sweep_error = ""
        if model.sweep is not None and len(model.groups) > 1:
            # V2-M4: task-level sweep (process pool, gpu_ids round-robin)
            results = driver.sweep_solve(prep, device=device or "")
            for r in results:
                _log(log, "sweep_task", group=r.group, gpu_id=r.gpu_id,
                     ok=r.ok, seconds=round(r.seconds, 3), error=r.error)
            sweep_error = driver.sweep_summary_error(results)
            metas = {r.group: r.meta for r in results
                     if r.ok and r.meta is not None}
            solved_groups = sorted(metas)
        else:
            metas = {}
            for g in solved_groups:
                t0 = time.time()
                metas[g] = driver.solve_group(prep, g, device=device or "",
                                              assembly=assembly or "")
                _log(log, "solve", group=g,
                     seconds=round(time.time() - t0, 3),
                     ndof=metas[g]["ndof"], residual=metas[g]["residual"])
        for g in solved_groups:
            meta = metas[g]
            _log(log, "solve_meta", group=g, ndof=meta["ndof"],
                 residual=meta["residual"])
            for pi, plane in enumerate(model.output.planes):
                path = outputs.write_plane_h5(prep, g, pi)
                _log(log, "plane", group=g, plane=pi, file=str(path))
            if model.output.orders_enabled and model.output.planes:
                # order tables for the top-most and bottom-most planes
                zs = [p.z for p in model.output.planes]
                top_i = int(max(range(len(zs)), key=lambda i: zs[i]))
                bot_i = int(min(range(len(zs)), key=lambda i: zs[i]))
                eps_top = model.eps_bg_of_slab(len(model.slabs) - 2)
                eps_bot = model.eps_bg_of_slab(0)
                h5p, csvp = outputs.write_orders_files(
                    prep, g, top_i, eps_top, +1,
                    outdir / f"orders_up_g{g}")
                _log(log, "orders", group=g, direction="up", h5=str(h5p))
                if bot_i != top_i:
                    h5p, csvp = outputs.write_orders_files(
                        prep, g, bot_i, eps_bot, -1,
                        outdir / f"orders_down_g{g}")
                    _log(log, "orders", group=g, direction="down", h5=str(h5p))
        _log(log, "done", seconds=round(time.time() - t_start, 3))
    if sweep_error:
        # failed tasks are isolated (successful groups were archived above);
        # surface the aggregate so callers/CI see the failure
        raise RuntimeError(sweep_error)
    return outdir


def cmd_selftest(full: bool) -> int:
    """Run the packaged acceptance tests (fast tier by default)."""
    tests_dir = Path(__file__).resolve().parents[2] / "tests"
    if not tests_dir.is_dir():
        print("selftest: bundled tests not found (source checkout required "
              "in v1; set LITHOFEM_TESTS to the tests directory)")
        return 2
    marker = [] if full else ["-m", "fast"]
    cmd = [sys.executable, "-m", "pytest", "-q", str(tests_dir), *marker]
    print("running:", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(tests_dir.parent))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="lithofem")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_run = sub.add_parser("run", help="run a simulation from a YAML config")
    ap_run.add_argument("config")
    ap_run.add_argument("-o", "--outdir", required=True)
    ap_run.add_argument("--device", choices=["cpu", "cuda"], default=None,
                        help="override solver.device from the config")
    ap_run.add_argument("-a", "--assembly", choices=["cpu", "gpu"],
                        default=None,
                        help="override fem.assembly from the config (v2.5 "
                        "GPU matrix assembly)")

    ap_st = sub.add_parser("selftest", help="run the packaged test suite")
    ap_st.add_argument("--full", action="store_true",
                       help="run the full tier (slow) instead of fast")

    args = ap.parse_args(argv)
    if args.cmd == "run":
        try:
            run(args.config, args.outdir, device=args.device,
                assembly=args.assembly)
        except Exception as e:  # noqa: BLE001 - single clean error surface
            from .config import ConfigError

            if isinstance(e, ConfigError):
                print("configuration error:", file=sys.stderr)
                for issue in e.issues:
                    print(f"  - {issue}", file=sys.stderr)
                return 2
            print(f"error: {e}", file=sys.stderr)
            return 1
        return 0
    return cmd_selftest(args.full)


if __name__ == "__main__":
    raise SystemExit(main())
