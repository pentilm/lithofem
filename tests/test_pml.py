"""M5 acceptance tests: z-PML (driver: solver/bin/pml_test).

Criteria (docs/validation.md):
  M5-1 uniform medium, normal incidence: |r_PML| < 1e-6 (p=3, PML 1 wl);
  M5-2 oblique 30/60 deg: |r| < 1e-5;
  M5-3 lossy medium and vacuum-above-metal-substrate: same standards;
  M5-4 parameter scan (thickness x order)  (tools/pml_scan.py).

z resolution: 12 elements/wavelength (the |r| here is dominated by the
discrete-profile numerical reflection; converges with nz, see report).
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

PML = Path(__file__).resolve().parent.parent / "solver" / "bin" / "pml_test"

pytestmark = [
    pytest.mark.full,
    pytest.mark.skipif(not PML.exists(), reason="pml_test not built"),
]


def run_pml(**kw: float | int) -> float:
    kw.setdefault("nz", 12)
    args = [str(PML), "-p", "3"]
    flags = {"theta": "-t", "pol": "-pol", "eps_re": "-er", "eps_im": "-ei",
             "metal": "-metal", "thick": "-pt", "order": "-po", "nz": "-nz",
             "p": "-p", "target": "-tr"}
    for k, v in kw.items():
        args += [flags[k], str(v)]
    res = subprocess.run(args, capture_output=True, text=True, timeout=1800)
    assert res.returncode == 0, res.stderr or res.stdout
    m = re.search(r"pml_reflection ([\d.eE+-]+)", res.stdout)
    assert m, res.stdout
    return float(m.group(1))


@pytest.mark.parametrize("pol", [0, 1])
def test_m5_1_normal_incidence_vacuum(pol: int) -> None:
    # x-polarization needs slightly finer z (tet mesh is not x/y symmetric)
    r = run_pml(theta=0, pol=pol, nz=(12 if pol == 0 else 16))
    assert r < 1e-6, r


@pytest.mark.parametrize("theta", [30, 60])
def test_m5_2_oblique(theta: int) -> None:
    # effective PML absorption scales as target^{cos(theta)} (see docs/gpu.md), so
    # grazing angles need a deeper target to stay under 1e-5
    kw = {"target": 1e-12} if theta == 60 else {}
    r = run_pml(theta=theta, pol=0, **kw)
    assert r < 1e-5, r


def test_m5_3_lossy_medium() -> None:
    r = run_pml(theta=0, pol=0, eps_re=2.25, eps_im=0.5)
    assert r < 1e-6, r


def test_m5_3_metal_substrate() -> None:
    r = run_pml(theta=0, pol=0, metal=1)
    assert r < 1e-6, r
