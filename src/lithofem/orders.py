"""Diffraction orders and Poynting fluxes from sampled plane fields
(docs/physics.md); used by M6b energy checks, formalized for M7).

Power quantities use the normalization P_tilde = Z0 * P (consistent across
all terms, so balances are unit-free).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Model
from .constants import k0 as vacuum_k0


@dataclass(frozen=True)
class OrderSet:
    """Per-order envelope coefficients on a z-plane and their wavevectors."""

    ms: np.ndarray        # (n,) x-order index
    ns: np.ndarray        # (n,) y-order index
    coeffs: np.ndarray    # (n, 3) complex E envelope coefficients
    kx: np.ndarray        # (n,) transverse wavevectors (incl. Bloch k)
    ky: np.ndarray
    qz: np.ndarray        # (n,) complex, Im >= 0
    k0: float
    eps: complex

    @property
    def propagating(self) -> np.ndarray:
        return np.abs(self.qz.imag) < 1e-12 * self.k0


def orders_from_plane(
    u: np.ndarray, model: Model, group: int, eps_medium: complex
) -> OrderSet:
    """FFT the (ny, nx, 3) envelope samples into Fourier orders."""
    ny, nx, _ = u.shape
    c = np.fft.fft2(u, axes=(0, 1)) / (nx * ny)
    k0 = vacuum_k0(model.wavelength)
    kbx, kby = model.groups[group].kpar
    gx = 2 * np.pi / model.domain.lx
    gy = 2 * np.pi / model.domain.ly
    ms, ns, coeffs, kxs, kys, qzs = [], [], [], [], [], []
    for j in range(ny):
        n_idx = j if j <= ny // 2 else j - ny
        for i in range(nx):
            m_idx = i if i <= nx // 2 else i - nx
            kx = kbx + gx * m_idx
            ky = kby + gy * n_idx
            qz2 = k0**2 * eps_medium - kx**2 - ky**2
            qz = np.sqrt(complex(qz2))
            if qz.imag < 0:
                qz = -qz
            ms.append(m_idx)
            ns.append(n_idx)
            coeffs.append(c[j, i])
            kxs.append(kx)
            kys.append(ky)
            qzs.append(qz)
    return OrderSet(
        ms=np.array(ms), ns=np.array(ns), coeffs=np.array(coeffs),
        kx=np.array(kxs), ky=np.array(kys), qz=np.array(qzs),
        k0=k0, eps=complex(eps_medium),
    )


def one_way_flux_z(orders: OrderSet, direction: int, area: float) -> float:
    """Normalized z-power of a one-way field (direction +1 up / -1 down).

    Each order is a plane wave with k = (kx, ky, direction*qz); the
    per-order flux density is  S_z = Re[(E x H*)_z] / 2  with
    Z0 H = k x E / k0; total power = sum over orders times the cell area
    (Parseval). Evanescent orders carry no real z-flux and drop out.
    """
    total = 0.0
    for i in range(len(orders.ms)):
        e = orders.coeffs[i]
        k = np.array([orders.kx[i], orders.ky[i], direction * orders.qz[i]])
        h = np.cross(k, e) / orders.k0  # Z0 * H
        s = np.cross(e, np.conj(h))
        total += 0.5 * float(np.real(s[2]))
    return total * area


def sheet_radiated_power(
    u_sheet: np.ndarray, model: Model, source_idx: int
) -> float:
    """Normalized power injected by a (horizontal, uniform-grid sampled)
    sheet source:  P = -Re( integral E . (Z0 J)* dA ) / 2.

    The envelope samples and the envelope current (Z0 J e^{i pg.r}) share the
    Bloch factor, which cancels inside E . J*.
    """
    from .config import SheetSource

    src = model.sources[source_idx]
    assert isinstance(src, SheetSource)
    ny, nx, _ = u_sheet.shape
    x = (np.arange(nx) + 0.5) * model.domain.lx / nx
    y = (np.arange(ny) + 0.5) * model.domain.ly / ny
    e1 = np.array(src.edges[0])
    e2 = np.array(src.edges[1])
    e1n = np.hypot(e1[0], e1[1])
    e2n = np.hypot(e2[0], e2[1])
    det = e1[0] * e2[1] - e1[1] * e2[0]
    rx = x[None, :] - src.corner[0]
    ry = y[:, None] - src.corner[1]
    s1 = (rx * e2[1] - ry * e2[0]) / det
    s2 = (e1[0] * ry - e1[1] * rx) / det
    inside = (s1 >= 0) & (s1 <= 1) & (s2 >= 0) & (s2 <= 1)
    phase = np.exp(1j * (src.phase_gradient[0] * s1 * e1n +
                         src.phase_gradient[1] * s2 * e2n))
    j_env = np.array(src.current)[None, None, :] * phase[..., None]
    integrand = np.sum(u_sheet * np.conj(j_env), axis=2) * inside
    da = (model.domain.lx / nx) * (model.domain.ly / ny)
    return -0.5 * float(np.real(integrand.sum() * da))


def absorbed_power(meta: dict, solve_json: dict) -> float:
    """Normalized absorption  (k0/2) * sum_regions Im(eps) * int |E|^2 dV."""
    k0 = solve_json["k0"]
    total = 0.0
    for k, reg in enumerate(solve_json["regions"]):
        if reg["kind"] in ("pml_bottom", "pml_top"):
            continue  # PML decay is accounted by the flux planes, not here
        im_eps = reg["epsilon"][1]
        if k < len(meta["region_l2sq"]) and im_eps != 0.0:
            total += 0.5 * k0 * im_eps * meta["region_l2sq"][k]
    return total
