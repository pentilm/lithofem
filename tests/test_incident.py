"""Consistency tests for the global-frame incident-field tables (M6 prep).

Each part of Ehat(z) must: satisfy the per-slab dispersion relation and
transversality (div E = 0); be tangentially continuous across interfaces;
reduce to the bare plane wave in a uniform background.
"""

from __future__ import annotations

import numpy as np
import pytest

from lithofem import config, incident
from lithofem.constants import k0 as vacuum_k0


def _model(layers: list, theta: float, phi: float, pol, frm: str = "top"):
    c = {
        "domain": {"Lx": 96, "Ly": 96, "z_min": 0, "z_max": 120},
        "materials": {"absorber": {"n": 0.95, "k": 0.031},
                      "oxide": {"epsilon": [2.13, 0.0]}},
        "layers": layers,
        "frustums": [{"vertices": [[24, 24], [72, 24], [72, 72], [24, 72]],
                      "z0": 0, "h": 60, "alpha": 85}],
        "wavelength": 13.5,
        "sources": [{"type": "planewave",
                     "incidence": {"theta": theta, "phi": phi, "from": frm},
                     "polarization": pol}],
    }
    return config.expand(c)


LAYERS = [{"z": [0, 60], "material": "absorber"}, {"z": [60, 70], "material": "oxide"}]

CASES = [
    (0.0, 0.0, "s", "top"), (6.0, 0.0, "s", "top"), (6.0, 0.0, "p", "top"),
    (17.0, 35.0, "s", "top"), (17.0, 35.0, "p", "top"),
    (6.0, 120.0, "p", "bottom"),
    (10.0, 45.0, {"jones": [[1, 0], [0, 1]]}, "top"),
]


@pytest.mark.fast
@pytest.mark.parametrize("theta,phi,pol,frm", CASES)
def test_dispersion_and_transversality(theta, phi, pol, frm) -> None:
    model = _model(LAYERS, theta, phi, pol, frm)
    inc = incident.group_incident(model, model.groups[0])
    k0 = vacuum_k0(model.wavelength)
    kx, ky = inc.kpar
    for slabs in inc.parts:
        for i, sw in enumerate(slabs):
            eps = model.eps_bg_of_slab(i)
            for q, v in ((sw.qA, sw.A), (sw.qB, sw.B)):
                if np.linalg.norm(v) < 1e-14:
                    continue
                disp = kx**2 + ky**2 + q**2 - k0**2 * eps
                assert abs(disp) < 1e-10 * k0**2, (i, disp)
                div = kx * v[0] + ky * v[1] + q * v[2]
                assert abs(div) < 1e-10 * k0 * np.linalg.norm(v), (i, div)


@pytest.mark.fast
@pytest.mark.parametrize("theta,phi,pol,frm", CASES)
def test_tangential_continuity(theta, phi, pol, frm) -> None:
    model = _model(LAYERS, theta, phi, pol, frm)
    inc = incident.group_incident(model, model.groups[0])
    h = 1e-9
    for zb in model.slabs[1:-1]:
        e_lo = inc.eval(np.array([zb - h]))[0]
        e_hi = inc.eval(np.array([zb + h]))[0]
        scale = max(np.linalg.norm(e_lo), 1e-12)
        assert np.linalg.norm(e_lo[:2] - e_hi[:2]) < 1e-6 * scale, zb


@pytest.mark.fast
@pytest.mark.parametrize("frm", ["top", "bottom"])
@pytest.mark.parametrize("pol", ["s", "p"])
def test_uniform_background_is_bare_wave(pol: str, frm: str) -> None:
    """Vacuum everywhere: no reflection, |Ehat| = 1, correct z phase."""
    model = _model([], 20.0, 30.0, pol, frm)
    inc = incident.group_incident(model, model.groups[0])
    assert abs(inc.r_amp) < 1e-12
    k0 = vacuum_k0(model.wavelength)
    kx, ky = inc.kpar
    qz = np.sqrt(k0**2 - kx**2 - ky**2)
    sz = -1.0 if frm == "top" else 1.0
    z = np.linspace(1.0, 119.0, 57)
    e = inc.eval(z)
    mags = np.linalg.norm(e, axis=1)
    assert np.max(np.abs(mags - 1.0)) < 1e-10
    # phase advance between successive z samples: e^{i sz qz dz}
    ratio = e[1:, :] / e[:-1, :]
    expected = np.exp(1j * sz * qz * (z[1] - z[0]))
    finite = np.abs(e[:-1, :]) > 1e-8
    assert np.max(np.abs(ratio[finite] - expected)) < 1e-8


@pytest.mark.fast
def test_energy_conservation_lossless_stack() -> None:
    """R + T = 1 for a lossless two-layer stack (meta amplitudes)."""
    layers = [{"z": [0, 60], "material": "oxide"}]
    model = _model(layers, 15.0, 0.0, "s")
    inc = incident.group_incident(model, model.groups[0])
    k0 = vacuum_k0(model.wavelength)
    kx, ky = inc.kpar
    # incidence ambient: top slab material = vacuum (uncovered 60..120);
    # exit ambient: bottom slab = oxide
    q_in = np.sqrt(k0**2 - kx**2 - ky**2)
    eps_out = 2.13
    q_out = np.sqrt(k0**2 * eps_out - kx**2 - ky**2)
    R = abs(inc.r_amp) ** 2
    T = abs(inc.t_amp) ** 2 * q_out / q_in
    assert abs(R + T - 1.0) < 1e-12


@pytest.mark.fast
def test_s_polarization_along_e_s_convention() -> None:
    """§1.3: normal incidence from top, s-pol => E along +y_hat, amp 1 at z_max."""
    model = _model([], 0.0, 0.0, "s", "top")
    inc = incident.group_incident(model, model.groups[0])
    e = inc.eval(np.array([model.domain.z_max - 1e-9]))[0]
    assert abs(e[1] - 1.0) < 1e-9, e
    assert abs(e[0]) < 1e-12 and abs(e[2]) < 1e-12


@pytest.mark.fast
def test_p_polarization_along_e_p_convention() -> None:
    """Oblique p-pol from top: E at entry ~ e_p = k_hat x e_s (unit amp)."""
    theta, phi = 30.0, 40.0
    model = _model([], theta, phi, "p", "top")
    inc = incident.group_incident(model, model.groups[0])
    e = inc.eval(np.array([model.domain.z_max - 1e-9]))[0]
    e_s, e_p, _ = incident._sp_basis(theta, phi, True)
    proj = np.vdot(e_p, e)  # component along e_p
    assert abs(abs(proj) - 1.0) < 1e-9, (proj, e)
    resid = e - proj * e_p
    assert np.linalg.norm(resid) < 1e-9
