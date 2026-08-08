"""1D multilayer transfer-matrix solver (analytic reference, docs/physics.md)).

This is the golden reference for incident-field loading and end-to-end
validation (M1). Requirements: arbitrary stacks, complex eps, arbitrary
incidence angle, s/p polarization, reflection/transmission coefficients and
the field distribution inside the stack, all at machine precision.

Conventions (constants.py): time factor e^{-i omega t}; lossy => Im(eps) > 0.
Forward-propagating/decaying waves along the local axis go like e^{+i q zeta}
with the branch Im(q) >= 0 (and Re(q) >= 0 when Im(q) == 0).

Local frame: the solver works in a 1D coordinate ``zeta`` increasing *along
the propagation direction*, with the incidence half-space at zeta <= 0, layer
interfaces at zeta = 0, d1, d1+d2, ..., and the exit half-space beyond. The
in-plane wavevector component kpar lies along the local x axis. Mapping to
the global lithofem frame (z axis, from: top/bottom) is done by callers.

Scalar unknown u: s-pol -> u = Ey; p-pol -> u = Hy_tilde = Z0 * Hy (E-field
units). Vector fields are reconstructed from u and du/dzeta.

Numerical scheme: per-layer plane-wave ansatz with *directional phase
referencing* (growing exponentials never appear), all interface-continuity
equations solved as one banded linear system. Unconditionally stable and
machine-precision accurate; O(N) layers is trivial at these sizes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import k0 as vacuum_k0

_BRANCH_TOL = 0.0


def kz_branch(eps: complex, k0: float, kpar: float) -> complex:
    """Longitudinal wavenumber q = sqrt(k0^2 eps - kpar^2), Im q >= 0 branch.

    For Im(q) == 0 the branch with Re(q) >= 0 is taken (propagating forward).
    """
    q = np.sqrt(complex(eps) * k0 * k0 - kpar * kpar + 0j)
    if q.imag < _BRANCH_TOL or (q.imag == 0.0 and q.real < 0.0):
        q = -q
    return complex(q)


@dataclass(frozen=True)
class Stack:
    """Multilayer stack in propagation order.

    eps_in: half-space the wave comes from; eps_out: exit half-space;
    eps: per-layer permittivities; d: per-layer thicknesses (nm), same length.
    """

    eps_in: complex
    eps_out: complex
    eps: tuple[complex, ...] = ()
    d: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if len(self.eps) != len(self.d):
            raise ValueError("eps and d must have equal length")
        if any(t <= 0 for t in self.d):
            raise ValueError("layer thicknesses must be positive")


@dataclass(frozen=True)
class TMMResult:
    """Solution record; amplitudes use directional phase referencing.

    In layer j (0=incidence ambient, 1..M layers, M+1=exit ambient):
        u_j(zeta) = a[j] * e^{+i q[j] (zeta - z_lo[j])}
                  + b[j] * e^{-i q[j] (zeta - z_hi[j])}
    with z_lo/z_hi the entrance-/exit-side boundaries of layer j
    (ambients: z_lo[0] = z_hi[0] = 0, z_lo[M+1] = z_hi[M+1] = total depth).
    a[0] is the incident amplitude (=1), b[0] = r, a[M+1] = t, b[M+1] = 0.
    """

    stack: Stack
    wavelength: float
    kpar: float
    pol: str
    r: complex
    t: complex
    a: np.ndarray
    b: np.ndarray
    q: np.ndarray
    z_lo: np.ndarray
    z_hi: np.ndarray

    @property
    def k0(self) -> float:
        return vacuum_k0(self.wavelength)

    # -- power coefficients -------------------------------------------------
    @property
    def R(self) -> float:
        return float(abs(self.r) ** 2)

    @property
    def T(self) -> float:
        """Power transmittance (flux ratio along the propagation axis)."""
        g_in = _gamma(self.pol, complex(self.stack.eps_in))
        g_out = _gamma(self.pol, complex(self.stack.eps_out))
        num = (self.q[-1] * g_out).real
        den = (self.q[0] * g_in).real
        return float(abs(self.t) ** 2 * num / den)

    # -- field evaluation ---------------------------------------------------
    def _layer_of(self, zeta: np.ndarray) -> np.ndarray:
        edges = self.z_lo[1:-1]  # interior entrance boundaries, ascending
        return np.searchsorted(np.concatenate((edges, self.z_hi[-1:])), zeta, side="right")

    def u(self, zeta: np.ndarray) -> np.ndarray:
        """Scalar field u(zeta): Ey (s) or Z0*Hy (p)."""
        zeta = np.asarray(zeta, dtype=float)
        j = self._layer_of(zeta)
        up = self.a[j] * np.exp(1j * self.q[j] * (zeta - self.z_lo[j]))
        dn = self.b[j] * np.exp(-1j * self.q[j] * (zeta - self.z_hi[j]))
        return np.asarray(up + dn)

    def du(self, zeta: np.ndarray) -> np.ndarray:
        """d u / d zeta."""
        zeta = np.asarray(zeta, dtype=float)
        j = self._layer_of(zeta)
        up = self.a[j] * np.exp(1j * self.q[j] * (zeta - self.z_lo[j]))
        dn = self.b[j] * np.exp(-1j * self.q[j] * (zeta - self.z_hi[j]))
        return np.asarray(1j * self.q[j] * (up - dn))

    def eps_at(self, zeta: np.ndarray) -> np.ndarray:
        zeta = np.asarray(zeta, dtype=float)
        j = self._layer_of(zeta)
        eps_all = np.concatenate(
            ([complex(self.stack.eps_in)], np.asarray(self.stack.eps, dtype=complex),
             [complex(self.stack.eps_out)])
        )
        return eps_all[j]

    def fields(self, zeta: np.ndarray) -> dict[str, np.ndarray]:
        """Vector fields in the local frame (kpar along local x).

        Returns E components and Z0*H components. s-pol: E = (0, u, 0),
        Z0*H = (-du/(i k0), 0, kpar u / k0). p-pol: Z0*H = (0, u, 0),
        E = (du/(i k0 eps), 0, -kpar u/(k0 eps)).
        """
        u = self.u(zeta)
        du = self.du(zeta)
        k0 = self.k0
        zeros = np.zeros_like(u)
        if self.pol == "s":
            return {
                "Ex": zeros, "Ey": u, "Ez": zeros,
                "Hx": -du / (1j * k0), "Hy": zeros, "Hz": self.kpar * u / k0,
            }
        eps = self.eps_at(zeta)
        return {
            "Ex": du / (1j * k0 * eps), "Ey": zeros, "Ez": -self.kpar * u / (k0 * eps),
            "Hx": zeros, "Hy": u, "Hz": zeros,
        }


def _gamma(pol: str, eps: complex) -> complex:
    """Interface weight: gamma * du/dzeta is continuous (1 for s, 1/eps for p)."""
    return 1.0 + 0.0j if pol == "s" else 1.0 / eps


def solve(
    stack: Stack,
    wavelength: float,
    *,
    theta_deg: float = 0.0,
    pol: str = "s",
    kpar: float | None = None,
) -> TMMResult:
    """Solve the multilayer problem for a unit-amplitude incident wave.

    theta_deg is the incidence angle in the *incidence half-space*, which must
    be lossless if theta_deg is used; alternatively pass kpar directly
    (rad/nm). pol is 's' or 'p'.
    """
    if pol not in ("s", "p"):
        raise ValueError(f"pol must be 's' or 'p', got {pol!r}")
    k0 = vacuum_k0(wavelength)
    if kpar is None:
        eps_in = complex(stack.eps_in)
        if abs(eps_in.imag) > 0:
            raise ValueError("theta_deg needs a lossless incidence medium; pass kpar instead")
        n_in = float(np.sqrt(eps_in.real))
        kpar = k0 * n_in * float(np.sin(np.deg2rad(theta_deg)))

    eps_all = np.concatenate(
        ([complex(stack.eps_in)], np.asarray(stack.eps, dtype=complex),
         [complex(stack.eps_out)])
    )
    m = len(stack.eps)  # number of finite layers
    q = np.array([kz_branch(e, k0, kpar) for e in eps_all])
    gam = np.array([_gamma(pol, e) for e in eps_all])

    depth = np.concatenate(([0.0], np.cumsum(stack.d)))  # interface positions
    z_lo = np.concatenate(([0.0], depth[:-1], depth[-1:]))
    z_hi = np.concatenate(([0.0], depth[1:], depth[-1:]))
    phi = np.exp(1j * q[1:-1] * np.asarray(stack.d))  # per-layer phase, |phi|<=1

    # Unknowns x = [b_0, a_1, b_1, ..., a_m, b_m, a_{m+1}], size 2m+2.
    n_unk = 2 * m + 2
    A = np.zeros((n_unk, n_unk), dtype=complex)
    rhs = np.zeros(n_unk, dtype=complex)

    def a_idx(j: int) -> int:  # index of a_j for j=1..m+1
        return 2 * j - 1

    def b_idx(j: int) -> int:  # index of b_j for j=0..m
        return 0 if j == 0 else 2 * j

    row = 0
    for i in range(m + 1):  # interface i between layer i and layer i+1
        jl, jr = i, i + 1
        # values/derivative factors of layer jl at its exit boundary
        if jl == 0:
            ul_a, ul_b = 1.0 + 0j, 1.0 + 0j  # a_0=1 known, b_0 unknown
        else:
            ul_a, ul_b = phi[jl - 1], 1.0 + 0j
        # values of layer jr at its entrance boundary
        if jr == m + 1:
            ur_a, ur_b = 1.0 + 0j, 0.0j  # b_{m+1}=0
        else:
            ur_a, ur_b = 1.0 + 0j, phi[jr - 1]

        # continuity of u
        if jl == 0:
            rhs[row] = -ul_a  # move known incident term to RHS
        else:
            A[row, a_idx(jl)] = ul_a
        A[row, b_idx(jl)] += ul_b
        A[row, a_idx(jr)] -= ur_a
        if jr <= m:
            A[row, b_idx(jr)] -= ur_b
        row += 1

        # continuity of gamma * du
        cl = 1j * q[jl] * gam[jl]
        cr = 1j * q[jr] * gam[jr]
        if jl == 0:
            rhs[row] = -cl * ul_a
        else:
            A[row, a_idx(jl)] = cl * ul_a
        A[row, b_idx(jl)] += -cl * ul_b
        A[row, a_idx(jr)] -= cr * ur_a
        if jr <= m:
            A[row, b_idx(jr)] -= -cr * ur_b
        row += 1

    x = np.linalg.solve(A, rhs)

    a = np.empty(m + 2, dtype=complex)
    b = np.empty(m + 2, dtype=complex)
    a[0] = 1.0
    b[0] = x[0]
    for j in range(1, m + 1):
        a[j] = x[a_idx(j)]
        b[j] = x[b_idx(j)]
    a[m + 1] = x[-1]
    b[m + 1] = 0.0

    return TMMResult(
        stack=stack, wavelength=wavelength, kpar=float(kpar), pol=pol,
        r=complex(b[0]), t=complex(a[m + 1]), a=a, b=b, q=q, z_lo=z_lo, z_hi=z_hi,
    )
