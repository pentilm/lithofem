"""M8-2: self-checks of the independent RCWA reference implementation.

- uniform-layer degeneration vs TMM < 1e-10 (measured: machine precision);
- lossless-grating energy conservation < 1e-12;
- mode-count convergence of the efficiencies (curve recorded in the M8
  acceptance report).
"""

from __future__ import annotations

import pytest

from lithofem import tmm

from .reference import rcwa


@pytest.mark.fast
@pytest.mark.parametrize("pol,tpol", [("TE", "s"), ("TM", "p")])
def test_uniform_degeneration_vs_tmm(pol: str, tpol: str) -> None:
    lay = [rcwa.Layer(120.0, (1.0,), (2.25 + 0.5j,)),
           rcwa.Layer(60.0, (1.0,), (1.9 + 0.1j,))]
    res = rcwa.solve(400.0, 193.0, 25.0, pol, lay, 1.0, 2.25, n_orders=11)
    st = tmm.Stack(1.0 + 0j, 2.25 + 0j, (2.25 + 0.5j, 1.9 + 0.1j),
                   (120.0, 60.0))
    ref = tmm.solve(st, 193.0, theta_deg=25.0, pol=tpol)
    h = 5
    assert abs(res.r_eff[h] - ref.R) < 1e-10
    assert abs(res.t_eff[h] - ref.T) < 1e-10


@pytest.mark.fast
@pytest.mark.parametrize("pol", ["TE", "TM"])
def test_lossless_energy_conservation(pol: str) -> None:
    lay = [rcwa.Layer(100.0, (0.5, 0.5), (2.25 + 0j, 1.0 + 0j))]
    res = rcwa.solve(400.0, 193.0, 10.0, pol, lay, 1.0, 2.25, n_orders=41)
    assert abs(res.r_eff.sum() + res.t_eff.sum() - 1.0) < 1e-12


@pytest.mark.fast
@pytest.mark.parametrize("pol", ["TE", "TM"])
def test_mode_count_convergence(pol: str) -> None:
    """Zeroth-order efficiency converges with retained orders (EUV absorber)."""
    eps_abs = complex(0.95, 0.031) ** 2
    lay = [rcwa.Layer(60.0, (0.5, 0.5), (eps_abs, 1.0 + 0j))]

    def eff0(n: int) -> float:
        res = rcwa.solve(48.0, 13.5, 6.0, pol, lay, 1.0, 1.0, n_orders=n)
        return float(res.r_eff[n // 2])

    vals = [eff0(n) for n in (21, 41, 81, 121)]
    diffs = [abs(vals[i + 1] - vals[i]) for i in range(3)]
    assert diffs[-1] < diffs[0] + 1e-15
    assert diffs[-1] < 1e-6, (vals, diffs)
