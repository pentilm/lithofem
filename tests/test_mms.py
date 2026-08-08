"""M4 acceptance tests: MMS verification of the complex curl-curl core.

Criteria (docs/validation.md):
  M4-1 h-convergence rate >= p - 0.2 for p = 1..4, for eps real/complex/tensor;
  M4-2 p = 6 runs and converges (arbitrary-order path);
  M4-3 assembled-matrix complex symmetry ||A - A^T||/||A|| < 1e-12;
  M4-4 direct-solve residual ||Ax-b||/||b|| < 1e-10.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import numpy as np
import pytest

MMS = Path(__file__).resolve().parent.parent / "solver" / "bin" / "mms_test"

SCHEDULES = {1: (4, 3), 2: (2, 3), 3: (2, 3), 4: (2, 3)}


def run_mms(p: int, eps: str, n0: int, levels: int) -> dict:
    res = subprocess.run(
        [str(MMS), "-p", str(p), "-e", eps, "-n", str(n0), "-l", str(levels)],
        capture_output=True, text=True, timeout=3600,
    )
    assert res.returncode == 0, res.stderr or res.stdout
    errs, hs = [], []
    sym = res_norm = None
    for line in res.stdout.splitlines():
        m = re.match(r"level \d+ ndof \d+ h ([\d.eE+-]+) err ([\d.eE+-]+)", line)
        if m:
            hs.append(float(m.group(1)))
            errs.append(float(m.group(2)))
        m = re.match(r"symmetry ([\d.eE+-]+)", line)
        if m:
            sym = float(m.group(1))
        m = re.match(r"residual ([\d.eE+-]+)", line)
        if m:
            res_norm = float(m.group(1))
    assert len(errs) == levels and sym is not None and res_norm is not None
    rate = float(np.log(errs[-2] / errs[-1]) / np.log(hs[-2] / hs[-1]))
    return {"errs": errs, "rate": rate, "symmetry": sym, "residual": res_norm}


needs_solver = pytest.mark.skipif(not MMS.exists(), reason="mms_test not built")


@needs_solver
@pytest.mark.full
@pytest.mark.parametrize("eps", ["real", "complex", "tensor"])
@pytest.mark.parametrize("p", [1, 2, 3, 4])
def test_convergence_rate(p: int, eps: str) -> None:
    n0, levels = SCHEDULES[p]
    out = run_mms(p, eps, n0, levels)
    assert out["rate"] >= p - 0.2, f"p={p} eps={eps}: rate {out['rate']:.3f}"
    assert out["symmetry"] < 1e-12, f"asymmetry {out['symmetry']:.2e}"
    assert out["residual"] < 1e-10, f"residual {out['residual']:.2e}"


@needs_solver
@pytest.mark.full
def test_p6_arbitrary_order() -> None:
    out = run_mms(6, "complex", 2, 2)
    # errors must decrease markedly under refinement (trend check)
    assert out["errs"][1] < 0.1 * out["errs"][0]
    assert out["residual"] < 1e-10
