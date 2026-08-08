"""Incident-field construction: TMM background solution in the global frame
(docs/physics.md), M6).

For each solve group (one Bloch k||) and each plane-wave source in it, the
background field is

    E_inc(r) = Ehat(z) * exp(i k|| . r||),

with Ehat piecewise  A exp(i qA (z - zA)) + B exp(i qB (z - zB))  per slab
(directional referencing: |exp| <= 1 inside the slab). This module computes
the per-slab complex vector amplitudes in *global* coordinates, summed over
all plane-wave sources of the group; the table goes into solve.json for the
C++ solver (exact evaluation, no interpolation) and is also evaluated here
for Python-side post-processing.

Conventions (§1.3): e^{-i omega t}; from: top = propagation toward -z;
e_s = (z_hat x k_hat)/|...| (normal incidence: e_s = y_hat), e_p = k_hat x e_s.
The TMM local frame: zeta increases along propagation, x_loc along k||.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import tmm
from .config import Model, PlaneWave, SolveGroup
from .constants import k0 as vacuum_k0


@dataclass(frozen=True)
class SlabWave:
    """Ehat(z) = A e^{i qA (z - zA)} + B e^{i qB (z - zB)} for z in [z_lo, z_hi]."""

    z_lo: float
    z_hi: float
    qA: complex
    zA: float
    A: np.ndarray  # complex (3,)
    qB: complex
    zB: float
    B: np.ndarray

    def eval(self, z: np.ndarray) -> np.ndarray:
        z = np.atleast_1d(np.asarray(z, dtype=float))
        up = np.exp(1j * self.qA * (z - self.zA))[:, None] * self.A[None, :]
        dn = np.exp(1j * self.qB * (z - self.zB))[:, None] * self.B[None, :]
        return up + dn


@dataclass(frozen=True)
class IncidentField:
    """Piecewise background field for one solve group.

    parts: one tuple of SlabWave (one per model slab) per plane-wave
    polarization component; evaluation sums the parts. Keeping parts
    separate is always exact (different components may carry different
    propagation references).
    """

    kpar: tuple[float, float]
    parts: tuple[tuple[SlabWave, ...], ...]
    part_sources: tuple[int, ...]  # source index per part
    r_amp: complex  # summed reflected amplitude at the incidence side (meta)
    t_amp: complex

    def eval(self, z: np.ndarray) -> np.ndarray:
        """Ehat(z) -> (n, 3) complex (no transverse phase factor)."""
        z = np.atleast_1d(np.asarray(z, dtype=float))
        out = np.zeros((len(z), 3), dtype=complex)
        for slabs in self.parts:
            edges = np.array([sw.z_lo for sw in slabs[1:]])
            idx = np.searchsorted(edges, z, side="right")  # unique slab per z
            for i, sw in enumerate(slabs):
                m = idx == i
                if m.any():
                    out[m] += sw.eval(z[m])
        return out


def _sp_basis(
    theta: float, phi: float, from_top: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Global-frame (e_s, e_p, k_hat) for the incident direction."""
    th = np.deg2rad(theta)
    ph = np.deg2rad(phi)
    sz = -1.0 if from_top else 1.0
    k_hat = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph),
                      sz * np.cos(th)])
    if theta == 0.0:
        e_s = np.array([0.0, 1.0, 0.0])
    else:
        e_s = np.cross([0.0, 0.0, 1.0], k_hat)
        e_s = e_s / np.linalg.norm(e_s)
    e_p = np.cross(k_hat, e_s)
    return e_s, e_p, k_hat


def _stack_for_group(model: Model) -> tuple[tmm.Stack, bool]:
    """TMM stack in propagation order for from_top (reversed for bottom)."""
    eps_slabs = [model.eps_bg_of_slab(i) for i in range(len(model.slabs) - 1)]
    d = [model.slabs[i + 1] - model.slabs[i] for i in range(len(model.slabs) - 1)]
    # ambients continue the outermost slab materials (PML continuation, §5)
    return (
        tmm.Stack(
            eps_in=eps_slabs[-1], eps_out=eps_slabs[0],
            eps=tuple(reversed(eps_slabs)), d=tuple(reversed(d)),
        ),
        True,
    )


@dataclass(frozen=True)
class _PWPart:
    slabs: tuple[SlabWave, ...]
    r_amp: complex
    t_amp: complex


def _planewave_field(model: Model, pw: PlaneWave, pol: str, amp: complex) -> _PWPart:
    """Per-slab global-frame field for one plane wave of pure s or p pol."""
    wl = model.wavelength
    k0 = vacuum_k0(wl)
    kpar_mag = float(np.hypot(*pw.kpar))
    e_s, e_p, _ = _sp_basis(pw.theta, pw.phi, pw.from_top)

    if pw.from_top:
        stack, _ = _stack_for_group(model)
        z_ref = model.slabs[-1]  # zeta = z_ref - z
    else:
        eps_slabs = [model.eps_bg_of_slab(i) for i in range(len(model.slabs) - 1)]
        d = [model.slabs[i + 1] - model.slabs[i] for i in range(len(model.slabs) - 1)]
        stack = tmm.Stack(eps_in=eps_slabs[0], eps_out=eps_slabs[-1],
                          eps=tuple(eps_slabs), d=tuple(d))
        z_ref = model.slabs[0]  # zeta = z - z_ref

    res = tmm.solve(stack, wl, kpar=kpar_mag, pol="s" if pol == "s" else "p")

    # local frame unit vectors in global coordinates. NOTE (§1.3 alignment):
    # x_loc is along k|| (TMM phase convention); y_loc = zeta_hat x x_loc is
    # then -e_s for from_top, so the s amplitude gets a sign flip below to
    # keep the field along +e_s. For p, TMM's u = Z0*Hy has |E| = |u|/n_in in
    # the ambient, so the amplitude is scaled by n_in to make the incident
    # E amplitude equal `amp` along e_p.
    ph = np.deg2rad(pw.phi)
    x_loc = np.array([np.cos(ph), np.sin(ph), 0.0])
    zeta_hat = np.array([0.0, 0.0, -1.0 if pw.from_top else 1.0])
    y_loc = np.cross(zeta_hat, x_loc)
    align = float(np.dot(y_loc, e_s))  # +-1: y_loc vs the §1.3 e_s direction
    if abs(align) < 0.5:  # pragma: no cover - safety
        raise RuntimeError("polarization basis mismatch")
    s_sign = float(np.sign(align))
    n_in = complex(np.sqrt(complex(stack.eps_in)))
    amp_field = amp * s_sign if pol == "s" else amp * n_in

    n_slab = len(model.slabs) - 1
    slabs: list[SlabWave] = []
    for i in range(n_slab):
        z_lo, z_hi = model.slabs[i], model.slabs[i + 1]
        # TMM layer index for this slab (finite layers are 1..m in tmm order)
        j = (n_slab - i) if pw.from_top else (i + 1)
        q = res.q[j]
        a, b = res.a[j], res.b[j]
        # tmm: u_j = a e^{i q (zeta - zeta_lo_j)} + b e^{-i q (zeta - zeta_hi_j)}
        z_zeta_lo = res.z_lo[j]
        z_zeta_hi = res.z_hi[j]
        eps_j = complex(res.eps_at(np.array([0.5 * (z_zeta_lo + z_zeta_hi)]))[0])

        # vector amplitudes in the local frame for the up (a) / down (b) parts
        if pol == "s":
            va_loc = np.array([0, 1, 0]) * a
            vb_loc = np.array([0, 1, 0]) * b
        else:
            # u = Z0 Hy; E = (du/(i k0 eps), 0, -kpar u/(k0 eps)) in local frame
            va_loc = np.array([res.q[j] / (k0 * eps_j), 0, -kpar_mag / (k0 * eps_j)]) * a
            vb_loc = np.array([-res.q[j] / (k0 * eps_j), 0, -kpar_mag / (k0 * eps_j)]) * b
        R = np.stack([x_loc, y_loc, zeta_hat], axis=1)  # local -> global
        va = amp_field * (R @ va_loc)
        vb = amp_field * (R @ vb_loc)

        # map zeta phases to global z:
        # from_top: zeta = z_ref - z  => e^{i q (zeta - zeta0)} = e^{-i q (z - (z_ref - zeta0))}
        if pw.from_top:
            slabs.append(SlabWave(
                z_lo=z_lo, z_hi=z_hi,
                qA=-q, zA=z_ref - z_zeta_lo, A=va,
                qB=q, zB=z_ref - z_zeta_hi, B=vb,
            ))
        else:
            slabs.append(SlabWave(
                z_lo=z_lo, z_hi=z_hi,
                qA=q, zA=z_ref + z_zeta_lo, A=va,
                qB=-q, zB=z_ref + z_zeta_hi, B=vb,
            ))
    # r_amp/t_amp in E-field s/p basis (§1.3): for p, TMM's r is already the
    # E-ratio (n cancels between incident and reflected in the same ambient);
    # t needs the n_in/n_out conversion from the Hy to the E normalization.
    del e_p
    if pol == "p":
        n_out = complex(np.sqrt(complex(stack.eps_out)))
        t_e = amp * res.t * n_in / n_out
    else:
        t_e = amp * res.t
    return _PWPart(slabs=tuple(slabs), r_amp=amp * res.r, t_amp=t_e)


def group_incident(model: Model, group: SolveGroup) -> IncidentField:
    """Background field of the group: one part per plane-wave pol component."""
    parts: list[_PWPart] = []
    part_sources: list[int] = []
    r_amp = 0j
    t_amp = 0j
    for idx in group.source_indices:
        src = model.sources[idx]
        if not isinstance(src, PlaneWave):
            continue
        if src.polarization in ("s", "p"):
            comps = [(src.polarization, src.amplitude)]
        else:  # jones: decompose transverse (Ex, Ey) onto (e_s, e_p)
            assert src.jones is not None
            e_s, e_p, _ = _sp_basis(src.theta, src.phi, src.from_top)
            jones = np.array([src.jones[0], src.jones[1]])
            m = np.array([[e_s[0], e_p[0]], [e_s[1], e_p[1]]])
            c_s, c_p = np.linalg.solve(m, jones)
            comps = [("s", src.amplitude * c_s), ("p", src.amplitude * c_p)]
        for pol, amp in comps:
            if amp == 0:
                continue
            part = _planewave_field(model, src, pol, amp)
            parts.append(part)
            part_sources.append(idx)
            r_amp += part.r_amp
            t_amp += part.t_amp
    return IncidentField(
        kpar=group.kpar,
        parts=tuple(p.slabs for p in parts),
        part_sources=tuple(part_sources),
        r_amp=r_amp, t_amp=t_amp,
    )


def incident_to_json(inc: IncidentField) -> list[dict]:
    def c(v: complex) -> list[float]:
        return [float(np.real(v)), float(np.imag(v))]

    return [
        {
            "source_index": inc.part_sources[i],
            "slabs": [
                {
                    "z": [sw.z_lo, sw.z_hi],
                    "qA": c(sw.qA), "zA": sw.zA, "A": [c(v) for v in sw.A],
                    "qB": c(sw.qB), "zB": sw.zB, "B": [c(v) for v in sw.B],
                }
                for sw in slabs
            ],
        }
        for i, slabs in enumerate(inc.parts)
    ]
