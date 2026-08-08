"""M1 acceptance tests: TMM analytic reference module.

Acceptance criteria (docs/validation.md):
  M1-1 single interface vs Fresnel, |dr|,|dt| < 1e-12 (oblique, s/p, lossy);
  M1-2 lossless multilayer |R+T-1| < 1e-12; fixed-seed 10-layer complex-eps
       reciprocity;
  M1-3 quarter-wave mirror peak reflectance vs analytic formula < 1e-10;
  M1-4 internal field vs independent numerical 1D Helmholtz solution,
       relative L2 error < 1e-8.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from lithofem import tmm
from lithofem.constants import k0 as vacuum_k0

from .data_gen import gen_tmm_cases


def fresnel(
    eps1: complex, eps2: complex, k0: float, kpar: float, pol: str
) -> tuple[complex, complex]:
    """Textbook Fresnel coefficients in the u-normalization of tmm.py.

    s: u = Ey;  r = (q1-q2)/(q1+q2), t = 2 q1/(q1+q2).
    p: u = Z0*Hy; r = (eps2 q1 - eps1 q2)/(eps2 q1 + eps1 q2),
       t = 2 eps2 q1/(eps2 q1 + eps1 q2).
    """
    q1 = tmm.kz_branch(eps1, k0, kpar)
    q2 = tmm.kz_branch(eps2, k0, kpar)
    if pol == "s":
        return (q1 - q2) / (q1 + q2), 2 * q1 / (q1 + q2)
    num_r = eps2 * q1 - eps1 * q2
    den = eps2 * q1 + eps1 * q2
    return num_r / den, 2 * eps2 * q1 / den


SINGLE_INTERFACE_CASES = [
    # (eps_in, eps_out, theta_deg) -- lossless, lossy, TIR-regime
    (1.0 + 0j, 2.25 + 0j, 0.0),
    (1.0 + 0j, 2.25 + 0j, 30.0),
    (1.0 + 0j, 2.25 + 0j, 75.0),
    (2.25 + 0j, 1.0 + 0j, 60.0),          # beyond critical angle (TIR)
    (1.0 + 0j, (0.95 + 0.031j) ** 2, 6.0),  # EUV absorber-like, lossy
    (1.0 + 0j, -20.0 + 1.5j, 45.0),       # metal-like (Re eps < 0)
    (2.13 + 0j, 1.5 + 0.4j, 40.0),
]


@pytest.mark.fast
@pytest.mark.parametrize("pol", ["s", "p"])
@pytest.mark.parametrize("eps1,eps2,theta", SINGLE_INTERFACE_CASES)
def test_fresnel_single_interface(eps1: complex, eps2: complex, theta: float, pol: str) -> None:
    wl = 193.0
    res = tmm.solve(tmm.Stack(eps_in=eps1, eps_out=eps2), wl, theta_deg=theta, pol=pol)
    r_ref, t_ref = fresnel(eps1, eps2, vacuum_k0(wl), res.kpar, pol)
    assert abs(res.r - r_ref) < 1e-12
    assert abs(res.t - t_ref) < 1e-12


@pytest.mark.fast
@pytest.mark.parametrize("pol", ["s", "p"])
@pytest.mark.parametrize("case", gen_tmm_cases.lossless_stacks())
def test_energy_conservation_lossless(case: dict, pol: str) -> None:
    stack = tmm.Stack(case["eps_in"], case["eps_out"], case["eps"], case["d"])
    res = tmm.solve(stack, case["wavelength"], theta_deg=case["theta_deg"], pol=pol)
    assert abs(res.R + res.T - 1.0) < 1e-12


@pytest.mark.fast
@pytest.mark.parametrize("pol", ["s", "p"])
def test_reciprocity_complex_stack(pol: str) -> None:
    """gamma_in q_in t_backward == gamma_out q_out t_forward (Wronskian invariant)."""
    case = gen_tmm_cases.complex_stack_10layer()
    fwd_stack = tmm.Stack(case["eps_in"], case["eps_out"], case["eps"], case["d"])
    bwd_stack = tmm.Stack(
        case["eps_out"], case["eps_in"], case["eps"][::-1], case["d"][::-1]
    )
    fwd = tmm.solve(fwd_stack, case["wavelength"], theta_deg=case["theta_deg"], pol=pol)
    bwd = tmm.solve(bwd_stack, case["wavelength"], kpar=fwd.kpar, pol=pol)
    lhs = fwd.q[0] * (1.0 if pol == "s" else 1.0 / case["eps_in"]) * bwd.t
    rhs = fwd.q[-1] * (1.0 if pol == "s" else 1.0 / case["eps_out"]) * fwd.t
    assert abs(lhs - rhs) / abs(rhs) < 1e-12


@pytest.mark.fast
def test_quarter_wave_mirror() -> None:
    """N-pair HL quarter-wave mirror at normal incidence vs analytic peak R."""
    wl = 550.0
    n_h, n_l, n_sub = 2.35, 1.46, 1.52
    n_pairs = 8
    eps: list[complex] = []
    d: list[float] = []
    for _ in range(n_pairs):
        eps += [complex(n_h**2), complex(n_l**2)]
        d += [wl / (4 * n_h), wl / (4 * n_l)]
    stack = tmm.Stack(1.0 + 0j, complex(n_sub**2), tuple(eps), tuple(d))
    res = tmm.solve(stack, wl, theta_deg=0.0, pol="s")
    # (HL)^N on substrate: admittance Y = (nH/nL)^(2N) * n_sub, R = ((1-Y)/(1+Y))^2
    y = (n_h / n_l) ** (2 * n_pairs) * n_sub
    r_analytic = ((1 - y) / (1 + y)) ** 2
    assert abs(res.R - r_analytic) < 1e-10


@pytest.mark.fast
@pytest.mark.parametrize("pol", ["s", "p"])
def test_interface_continuity(pol: str) -> None:
    """u and gamma*du continuous across every interface (self-consistency)."""
    case = gen_tmm_cases.complex_stack_10layer()
    stack = tmm.Stack(case["eps_in"], case["eps_out"], case["eps"], case["d"])
    res = tmm.solve(stack, case["wavelength"], theta_deg=case["theta_deg"], pol=pol)
    eps_all = [case["eps_in"], *case["eps"], case["eps_out"]]
    z = np.cumsum([0.0, *case["d"]])
    h = 1e-9
    for i, zi in enumerate(z):
        gl = 1.0 if pol == "s" else 1.0 / eps_all[i]
        gr = 1.0 if pol == "s" else 1.0 / eps_all[i + 1]
        ul, ur = res.u(np.array([zi - h]))[0], res.u(np.array([zi + h]))[0]
        dl, dr = res.du(np.array([zi - h]))[0], res.du(np.array([zi + h]))[0]
        assert abs(ul - ur) < 1e-6 * abs(ur) + 1e-9
        assert abs(gl * dl - gr * dr) < 1e-6 * abs(gr * dr) + 1e-9


def _ode_reference_u(
    stack: tmm.Stack, wavelength: float, kpar: float, pol: str, z_eval: np.ndarray
) -> np.ndarray:
    """Independent 1D Helmholtz solution by high-order ODE integration (shooting).

    Integrates u'' = -(k0^2 eps - kpar^2) u (s-pol), or the equivalent
    first-order system in (u, w = gamma u') for p-pol, layer by layer from the
    exit side (pure outgoing wave) backwards, applying interface continuity of
    (u, gamma u'). The incident amplitude is then read off at the entrance and
    the whole solution rescaled. Uses DOP853 with tight tolerances; completely
    independent of the analytic per-layer exponentials in tmm.solve.
    """
    k0 = vacuum_k0(wavelength)
    eps_all = [complex(stack.eps_in), *stack.eps, complex(stack.eps_out)]
    z = np.cumsum([0.0, *stack.d])
    total = z[-1]
    q_out = tmm.kz_branch(eps_all[-1], k0, kpar)
    g_out = 1.0 if pol == "s" else 1.0 / eps_all[-1]

    # state y = [Re u, Im u, Re w, Im w], w = gamma * u'
    # ODE system: u' = w / gamma ; w' = -gamma (k0^2 eps - kpar^2) u.
    def rhs_factory(eps: complex):  # noqa: ANN202
        gam = 1.0 if pol == "s" else 1.0 / eps
        cc = -gam * (k0 * k0 * eps - kpar * kpar)

        def rhs(_t: float, y: np.ndarray) -> np.ndarray:
            u = y[0] + 1j * y[1]
            w = y[2] + 1j * y[3]
            du = w / gam
            dw = cc * u
            return np.array([du.real, du.imag, dw.real, dw.imag])

        return rhs

    # start at exit boundary: u = 1, w = gamma_out * i q_out
    u0 = 1.0 + 0j
    w0 = g_out * 1j * q_out * u0
    state = np.array([u0.real, u0.imag, w0.real, w0.imag])

    # store per-layer dense solutions for evaluation
    sols = []
    for j in range(len(eps_all) - 2, 0, -1):  # finite layers, exit side first
        sol = solve_ivp(
            rhs_factory(eps_all[j]),
            (z[j], z[j - 1]),
            state,
            method="DOP853",
            rtol=1e-13,
            atol=1e-13,
            dense_output=True,
        )
        assert sol.success
        sols.append((j, sol))
        state = sol.y[:, -1]

    u_top = state[0] + 1j * state[1]
    w_top = state[2] + 1j * state[3]
    q_in = tmm.kz_branch(eps_all[0], k0, kpar)
    g_in = 1.0 if pol == "s" else 1.0 / eps_all[0]
    du_top = w_top / g_in
    a_inc = (1j * q_in * u_top + du_top) / (2j * q_in)  # incident amplitude at z=0

    out = np.empty(len(z_eval), dtype=complex)
    for i, ze in enumerate(z_eval):
        if ze <= 0.0:
            u_val = u_top  # only used at boundary in tests
        elif ze >= total:
            u_val = np.exp(1j * q_out * (ze - total))
        else:
            j = int(np.searchsorted(z, ze, side="right"))  # layer index
            sol = next(s for jj, s in sols if jj == j)
            y = sol.sol(ze)
            u_val = y[0] + 1j * y[1]
        out[i] = u_val / a_inc
    return out


@pytest.mark.fast
@pytest.mark.parametrize("pol", ["s", "p"])
def test_internal_field_vs_ode(pol: str) -> None:
    case = gen_tmm_cases.lossy_field_stack()
    stack = tmm.Stack(case["eps_in"], case["eps_out"], case["eps"], case["d"])
    res = tmm.solve(stack, case["wavelength"], theta_deg=case["theta_deg"], pol=pol)
    total = sum(case["d"])
    z_eval = np.linspace(0.5, total - 0.5, 400)
    u_tmm = res.u(z_eval)
    u_ode = _ode_reference_u(stack, case["wavelength"], res.kpar, pol, z_eval)
    rel_l2 = np.linalg.norm(u_tmm - u_ode) / np.linalg.norm(u_ode)
    assert rel_l2 < 1e-8
