"""Output post-processing (M7): HDF5 z-slice fields with metadata,
diffraction-order files (HDF5 + CSV), energy-conservation report.

Physical fields on a plane:
    E_total = (u + Ehat_inc(z)) e^{i kpar.r}
    Z0*H    = (curl(E))/(i k0);  curl(E_sc) = e^{i kpar.r} (curl u + i K u)
with curl u sampled by the solver and K = kpar x. The incident H comes from
the per-part plane-wave representation analytically.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import h5py
import numpy as np

from . import driver, incident, orders
from .config import Model, PlaneWave
from .constants import ANGLE_UNIT, LENGTH_UNIT, TIME_CONVENTION
from .constants import k0 as vacuum_k0


def load_plane_curl(prep: driver.Prepared, group: int, plane: int) -> np.ndarray:
    p = prep.model.output.planes[plane]
    nx, ny = p.resolution
    raw = np.fromfile(
        prep.workdir / f"plane_g{group}_p{plane}_curl.bin", dtype=np.float64
    ).reshape(ny, nx, 3, 2)
    return raw[..., 0] + 1j * raw[..., 1]


def _incident_EH(model: Model, group: int, z: float) -> tuple[np.ndarray, np.ndarray]:
    """Ehat_inc(z) and Z0*Hhat_inc(z) (3,) complex, envelope (no kpar phase)."""
    inc = incident.group_incident(model, model.groups[group])
    k0 = vacuum_k0(model.wavelength)
    kx, ky = model.groups[group].kpar
    e = np.zeros(3, dtype=complex)
    h = np.zeros(3, dtype=complex)
    for slabs in inc.parts:
        for sw in slabs:
            if not (sw.z_lo - 1e-9 <= z <= sw.z_hi + 1e-9):
                continue
            for q, ref_z, amp in ((sw.qA, sw.zA, sw.A), (sw.qB, sw.zB, sw.B)):
                if np.linalg.norm(amp) < 1e-300:
                    continue
                ph = np.exp(1j * q * (z - ref_z))
                kvec = np.array([kx, ky, q])
                e += amp * ph
                h += np.cross(kvec, amp) / k0 * ph
            break
    return e, h


def plane_fields(
    prep: driver.Prepared, group: int, plane: int
) -> dict[str, np.ndarray]:
    """Physical total/scattered E and Z0*H on the sample grid."""
    model = prep.model
    z = model.output.planes[plane].z
    k0 = vacuum_k0(model.wavelength)
    kx, ky = model.groups[group].kpar
    u = driver.load_plane_envelope(prep, group, plane)
    cu = load_plane_curl(prep, group, plane)
    x, y = driver.plane_grid(model, plane)
    phase = np.exp(1j * (ky * y[:, None] + kx * x[None, :]))[..., None]

    e_sc = u * phase
    kvec = np.array([kx, ky, 0.0])
    ku = np.cross(np.broadcast_to(kvec, u.shape), u) * 1j
    h_sc = (cu + ku) / (1j * k0) * phase

    e_inc_hat, h_inc_hat = _incident_EH(model, group, z)
    e_tot = e_sc + e_inc_hat[None, None, :] * phase
    h_tot = h_sc + h_inc_hat[None, None, :] * phase
    return {"E": e_tot, "H": h_tot, "E_sc": e_sc, "H_sc": h_sc}


def write_plane_h5(prep: driver.Prepared, group: int, plane: int,
                   out_path: str | Path | None = None) -> Path:
    """HDF5 z-slice with complete metadata (docs/physics.md)."""
    model = prep.model
    p = model.output.planes[plane]
    fields = plane_fields(prep, group, plane)
    x, y = driver.plane_grid(model, plane)
    src_meta = []
    for s in model.sources:
        if isinstance(s, PlaneWave):
            src_meta.append({
                "type": "planewave", "theta_deg": s.theta, "phi_deg": s.phi,
                "from": "top" if s.from_top else "bottom",
                "polarization": s.polarization,
                "amplitude": [s.amplitude.real, s.amplitude.imag],
            })
        else:
            src_meta.append({"type": type(s).__name__.lower()})
    out = Path(out_path) if out_path else prep.workdir / f"g{group}_{p.file}"
    with h5py.File(out, "w") as f:
        for q in p.quantities:
            arr = fields[q]
            f.create_dataset(f"{q}_re", data=arr.real, track_times=False)
            f.create_dataset(f"{q}_im", data=arr.imag, track_times=False)
        f.create_dataset("x", data=x, track_times=False)
        f.create_dataset("y", data=y, track_times=False)
        f.attrs["z"] = p.z
        f.attrs["wavelength_nm"] = model.wavelength
        f.attrs["k0_rad_per_nm"] = vacuum_k0(model.wavelength)
        f.attrs["time_convention"] = TIME_CONVENTION
        f.attrs["length_unit"] = LENGTH_UNIT
        f.attrs["angle_unit"] = ANGLE_UNIT
        f.attrs["mu_r"] = 1.0
        f.attrs["H_normalization"] = "Z0*H (E-field units)"
        f.attrs["bloch_kpar_rad_per_nm"] = list(model.groups[group].kpar)
        f.attrs["group"] = group
        f.attrs["sources_json"] = json.dumps(src_meta)
        f.attrs["field_layout"] = "(ny, nx, 3) complex split re/im"
    return out


def _sp_reflected_basis(kx: float, ky: float, qz: complex, k0: float,
                        up: bool) -> tuple[np.ndarray, np.ndarray]:
    """(e_s, e_p) for an order propagating up (+z) or down (per §1.3)."""
    kz = qz.real if up else -qz.real
    k_hat = np.array([kx, ky, kz]) / k0
    kt = np.hypot(kx, ky)
    if kt < 1e-14 * k0:
        e_s = np.array([0.0, 1.0, 0.0])
    else:
        e_s = np.array([-ky, kx, 0.0]) / kt
    e_p = np.cross(k_hat, e_s)
    return e_s, e_p


def write_orders_files(
    prep: driver.Prepared, group: int, plane: int, eps_medium: complex,
    direction: int, out_stem: str | Path | None = None,
    include_incident_00: bool = True,
) -> tuple[Path, Path]:
    """Diffraction-order table -> HDF5 + CSV (docs/physics.md, M7).

    The plane must lie in a one-way region (above the structure for
    reflection with direction=+1, below for transmission with -1). The
    scattered orders are propagated to their reference at the domain top /
    bottom; the (0,0) order optionally gains the analytic incident/reflected
    (or transmitted) background amplitude.
    """
    model = prep.model
    z = model.output.planes[plane].z
    u = driver.load_plane_envelope(prep, group, plane)
    oset = orders.orders_from_plane(u, model, group, eps_medium)
    z_ref = model.domain.z_max if direction > 0 else model.domain.z_min
    inc = incident.group_incident(model, model.groups[group])

    rows = []
    area = model.domain.lx * model.domain.ly
    for i in range(len(oset.ms)):
        c = oset.coeffs[i] * np.exp(-1j * direction * oset.qz[i] * (z - z_ref))
        if include_incident_00 and oset.ms[i] == 0 and oset.ns[i] == 0:
            amp00 = inc.r_amp if direction > 0 else inc.t_amp
            e_s, e_p = _sp_reflected_basis(
                oset.kx[i], oset.ky[i], oset.qz[i], oset.k0, direction > 0)
            # background amplitude is stored in the s/p basis of the
            # reflected/transmitted wave; project it onto that basis
            pol = None
            for si in model.groups[group].source_indices:
                s = model.sources[si]
                if isinstance(s, PlaneWave):
                    pol = s.polarization
                    break
            if pol == "s":
                c = c + amp00 * e_s
            elif pol == "p":
                c = c + amp00 * e_p
        e_s, e_p = _sp_reflected_basis(
            oset.kx[i], oset.ky[i], oset.qz[i], oset.k0, direction > 0)
        a_s = complex(np.dot(e_s, c))
        a_p = complex(np.dot(e_p, c))
        kvec = np.array([oset.kx[i], oset.ky[i], direction * oset.qz[i]])
        h = np.cross(kvec, c) / oset.k0
        flux = 0.5 * float(np.real(np.cross(c, np.conj(h))[2])) * area
        prop = bool(abs(oset.qz[i].imag) < 1e-12 * oset.k0)
        rows.append({
            "m": int(oset.ms[i]), "n": int(oset.ns[i]),
            "kx": float(oset.kx[i]), "ky": float(oset.ky[i]),
            "kz_re": float((direction * oset.qz[i]).real),
            "kz_im": float((direction * oset.qz[i]).imag),
            "propagating": prop,
            "amp_s_re": a_s.real, "amp_s_im": a_s.imag,
            "amp_p_re": a_p.real, "amp_p_im": a_p.imag,
            "Ex_re": c[0].real, "Ex_im": c[0].imag,
            "Ey_re": c[1].real, "Ey_im": c[1].imag,
            "Ez_re": c[2].real, "Ez_im": c[2].imag,
            "flux_z": flux,
        })

    stem = Path(out_stem) if out_stem else prep.workdir / f"orders_g{group}_p{plane}"
    h5_path = stem.with_suffix(".h5")
    csv_path = stem.with_suffix(".csv")
    keys = list(rows[0].keys())
    with h5py.File(h5_path, "w") as f:
        for k in keys:
            f.create_dataset(k, data=np.array([r[k] for r in rows]),
                             track_times=False)
        f.attrs["wavelength_nm"] = model.wavelength
        f.attrs["direction"] = direction
        f.attrs["z_ref"] = z_ref
        f.attrs["eps_medium"] = [eps_medium.real, eps_medium.imag]
        f.attrs["time_convention"] = TIME_CONVENTION
        f.attrs["basis"] = "s/p per §1.3 for the order's own direction"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    return h5_path, csv_path


def order_efficiencies(
    prep: driver.Prepared, group: int, plane: int, eps_medium: complex,
    direction: int,
) -> dict[tuple[int, int], float]:
    """Order efficiencies (flux / incident flux) for a one-way region.

    direction=+1: reflection side (total up-going = scattered + analytic
    (0,0) reflected); -1: transmission side (scattered + analytic (0,0)
    transmitted). Incident flux from the group's plane-wave source(s),
    assumed unit-amplitude single polarization (the standard test setup).
    """
    model = prep.model
    z = model.output.planes[plane].z
    u = driver.load_plane_envelope(prep, group, plane)
    oset = orders.orders_from_plane(u, model, group, eps_medium)
    inc = incident.group_incident(model, model.groups[group])
    k0 = vacuum_k0(model.wavelength)
    kx, ky = model.groups[group].kpar

    # incidence-side medium = top slab background (continuation)
    eps_in = model.eps_bg_of_slab(len(model.slabs) - 2)
    qz_in = np.sqrt(complex(k0**2 * eps_in - kx**2 - ky**2))
    inc_flux = 0.5 * (qz_in.real / k0) * model.domain.lx * model.domain.ly

    pol = None
    for si in model.groups[group].source_indices:
        s = model.sources[si]
        if isinstance(s, PlaneWave):
            pol = s.polarization
            break

    area = model.domain.lx * model.domain.ly
    z_ref = model.domain.z_max if direction > 0 else model.domain.z_min
    out: dict[tuple[int, int], float] = {}
    for i in range(len(oset.ms)):
        # propagate the scattered coefficient to the reference plane so the
        # analytic (0,0) background term (referenced there) adds in phase
        c = oset.coeffs[i] * np.exp(-1j * direction * oset.qz[i] * (z - z_ref))
        if oset.ms[i] == 0 and oset.ns[i] == 0:
            amp00 = inc.r_amp if direction > 0 else inc.t_amp
            e_s, e_p = _sp_reflected_basis(
                oset.kx[i], oset.ky[i], oset.qz[i], oset.k0, direction > 0)
            if pol == "s":
                c = c + amp00 * e_s
            elif pol == "p":
                c = c + amp00 * e_p
        kvec = np.array([oset.kx[i], oset.ky[i], direction * oset.qz[i]])
        h = np.cross(kvec, c) / oset.k0
        flux = 0.5 * float(np.real(np.cross(c, np.conj(h))[2])) * area
        out[(int(oset.ms[i]), int(oset.ns[i]))] = direction * flux / inc_flux
    return out
