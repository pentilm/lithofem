"""Independent 1D RCWA reference implementation (M8 parallel deliverable).

Lamellar (binary, y-invariant) gratings, TE/TM, arbitrary layer stacks,
oblique incidence in the xz-plane. Uses Li's inverse-rule Fourier
factorization for TM and a Redheffer S-matrix recursion for stability.

Conventions match lithofem (§1.3): e^{-i omega t}, lossy Im(eps) > 0,
incidence from the top ambient propagating toward -z; the grating is
periodic in x with period Lambda. Orders m carry kx_m = kx0 + m 2pi/Lambda.

Self-checks required by M8-2: uniform-layer degeneration vs TMM to 1e-10
and mode-count convergence (see tests).

This module is intentionally independent of lithofem.tmm internals: it only
shares the physical conventions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eig, toeplitz


@dataclass(frozen=True)
class Layer:
    """One z-slab: piecewise-constant eps(x) given as (fractions, eps) bins."""

    thickness: float
    fractions: tuple[float, ...]  # widths as fractions of the period, sum 1
    eps: tuple[complex, ...]


@dataclass(frozen=True)
class RcwaResult:
    orders: np.ndarray        # (M,) order indices
    r: np.ndarray             # (M,) complex reflection amplitudes (Ey or Hy)
    t: np.ndarray             # (M,) complex transmission amplitudes
    r_eff: np.ndarray         # (M,) reflected efficiencies
    t_eff: np.ndarray         # (M,) transmitted efficiencies
    kx: np.ndarray
    qz_in: np.ndarray         # (M,) kz/k0 in the incidence ambient
    qz_out: np.ndarray


def _fourier_coeffs(fractions, eps_vals, orders_needed):
    """Fourier coefficients of a piecewise-constant periodic function."""
    edges = np.concatenate(([0.0], np.cumsum(fractions)))
    coeffs = {}
    for p in range(-orders_needed, orders_needed + 1):
        total = 0.0 + 0.0j
        for (a, b), e in zip(zip(edges[:-1], edges[1:]), eps_vals):
            if p == 0:
                total += e * (b - a)
            else:
                total += e * (np.exp(-2j * np.pi * p * b) -
                              np.exp(-2j * np.pi * p * a)) / (-2j * np.pi * p)
        coeffs[p] = total
    return coeffs


def _toeplitz_from(coeffs, m):
    col = np.array([coeffs[p] for p in range(0, m)])
    row = np.array([coeffs[-p] for p in range(0, m)])
    return toeplitz(col, row)


def _layer_modes(layer: Layer, kxn: np.ndarray, pol: str):
    """Eigenmodes of one layer: (W, V, q) with fields ~ e^{+- k0 q z}."""
    m = len(kxn)
    kx = np.diag(kxn)
    ec = _fourier_coeffs(layer.fractions, layer.eps, m)
    E = _toeplitz_from(ec, m)
    def branch(q):
        # forward modes ~ e^{-k0 q z} with z increasing INTO the stack:
        # decaying (Re q > 0); pure propagating -> q = -i|qz| (downward,
        # e^{+i k0 qz z} under e^{-i omega t})
        q = np.where(q.real < 0, -q, q)
        return np.where((np.abs(q.real) < 1e-14) & (q.imag > 0), -q, q)

    if pol == "TE":
        # d2Sy/dz~2 = (Kx^2 - E) Sy;  V = W Q (second tangential field Hx)
        w, W = eig(kx @ kx - E)
        q = branch(np.sqrt(w))
        V = W @ np.diag(q)
        return W, V, q
    # TM with Li's inverse rule:
    #   d2U/dz~2 = E_li (Kx E^{-1} Kx - I) U,  E_li = inv(toeplitz(1/eps))
    #   Ex-like field: V = E_li^{-1} W Q = P W Q
    inv_ec = _fourier_coeffs(layer.fractions, tuple(1.0 / np.asarray(layer.eps)),
                             m)
    P = _toeplitz_from(inv_ec, m)
    E_li = np.linalg.inv(P)
    a_mat = E_li @ (kx @ np.linalg.solve(E, kx) - np.eye(m))
    w, W = eig(a_mat)
    q = branch(np.sqrt(w))
    V = P @ W @ np.diag(q)
    return W, V, q


def _homog_modes(eps: complex, kxn: np.ndarray, pol: str):
    """Analytic modes of a homogeneous layer (also the ambient basis).

    Same conventions as _layer_modes: forward modes ~ e^{-k0 q z} with
    q = i qz (downward propagation), V = W Q (TE) or (1/eps) W Q (TM).
    Returns (W, V, q, qz).
    """
    qz = np.sqrt(eps - kxn**2 + 0j)
    qz = np.where(qz.imag < 0, -qz, qz)
    qz = np.where((qz.imag == 0) & (qz.real < 0), -qz, qz)
    q = -1j * qz  # forward = downward: e^{-k0 q z} = e^{+i k0 qz z}
    W = np.eye(len(kxn))
    V = np.diag(q) if pol == "TE" else np.diag(q / eps)
    return W, V, q, qz


def solve(
    period: float,
    wavelength: float,
    theta_deg: float,
    pol: str,
    layers: list[Layer],
    eps_in: complex = 1.0,
    eps_out: complex = 1.0,
    n_orders: int = 21,
) -> RcwaResult:
    """Solve the grating problem; n_orders = total retained orders (odd)."""
    assert pol in ("TE", "TM")
    assert n_orders % 2 == 1
    half = n_orders // 2
    k0 = 2 * np.pi / wavelength
    n_in = np.sqrt(complex(eps_in)).real
    kx0 = n_in * np.sin(np.deg2rad(theta_deg))
    ms = np.arange(-half, half + 1)
    kxn = kx0 + ms * wavelength / period  # normalized kx/k0

    w_in, v_in, _, qz_in = _homog_modes(complex(eps_in), kxn, pol)
    w_out, v_out, _, qz_out = _homog_modes(complex(eps_out), kxn, pol)

    m = n_orders
    ident = np.eye(m)

    # Redheffer star: accumulate S = [[S11, S12], [S21, S22]] from the top
    # ambient down through all layers to the bottom ambient. Mode amplitudes:
    # incoming from top a, outgoing up r = S11 a; transmitted t = S21 a.
    s11, s12 = np.zeros((m, m), complex), ident.copy()
    s21, s22 = ident.copy(), np.zeros((m, m), complex)

    def star(sA, sB):
        a11, a12, a21, a22 = sA
        b11, b12, b21, b22 = sB
        inv1 = np.linalg.solve(ident - a22 @ b11, a21)
        inv2 = np.linalg.solve(ident - b11 @ a22, b12)
        c11 = a11 + a12 @ b11 @ inv1
        c12 = a12 @ inv2
        c21 = b21 @ inv1
        c22 = b22 + b21 @ a22 @ inv2
        return c11, c12, c21, c22

    def interface_s(W1, V1, W2, V2):
        """S-matrix of the bare interface between mode bases 1 (top) and 2."""
        # continuity of the two tangential field components; solve the
        # block system directly for transmitted/reflected mode amplitudes
        M = np.block([[W2, -W1], [V2, V1]])
        N = np.vstack([W1, V1])
        sol = np.linalg.solve(M, N)          # [t; r] for incidence from top
        t_tb = sol[:m]
        r_tb = sol[m:]
        M2 = np.block([[W1, -W2], [-V1, -V2]])
        N2 = np.vstack([W2, -V2])
        sol2 = np.linalg.solve(M2, N2)       # incidence from bottom
        t_bt = sol2[:m]
        r_bt = sol2[m:]
        return r_tb, t_bt, t_tb, r_bt        # (S11, S12, S21, S22)

    def prop_s(q, d):
        X = np.diag(np.exp(-k0 * q * d))
        z = np.zeros((m, m), complex)
        return z, X, X, z

    S = (s11, s12, s21, s22)
    prev = (w_in, v_in)
    for lay in layers:
        if len(set(lay.eps)) == 1:
            W, V, q, _ = _homog_modes(complex(lay.eps[0]), kxn, pol)
        else:
            W, V, q = _layer_modes(lay, kxn, pol)
        S = star(S, interface_s(prev[0], prev[1], W, V))
        S = star(S, prop_s(q, lay.thickness))
        prev = (W, V)
    S = star(S, interface_s(prev[0], prev[1], w_out, v_out))

    # incidence: unit amplitude in order 0 mode of the top ambient
    a_in = np.zeros(m, complex)
    a_in[half] = 1.0
    r = S[0] @ a_in
    t = S[2] @ a_in

    qz0 = qz_in[half]
    if pol == "TE":
        r_eff = np.abs(r) ** 2 * np.real(qz_in) / qz0.real
        t_eff = np.abs(t) ** 2 * np.real(qz_out) / qz0.real
    else:
        r_eff = np.abs(r) ** 2 * np.real(qz_in / complex(eps_in)) / \
            np.real(qz0 / complex(eps_in))
        t_eff = np.abs(t) ** 2 * np.real(qz_out / complex(eps_out)) / \
            np.real(qz0 / complex(eps_in))
    return RcwaResult(orders=ms, r=r, t=t, r_eff=r_eff, t_eff=t_eff,
                      kx=kxn, qz_in=qz_in, qz_out=qz_out)
