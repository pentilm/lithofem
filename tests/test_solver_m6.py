"""M6 acceptance tests: Bloch periodicity + scattered-field formulation,
end-to-end against TMM.

Criteria (docs/validation.md):
  M6-1 pure multilayer: scattered energy / incident energy < 1e-8;
  M6-2 artificial difference layer: total field vs full-stack TMM on the
       observation plane, rel L2 < 1e-4 (p=3), monotone in p=2,3,4;
       theta in {6, 17} deg, s & p, EUV (13.5) & DUV (193);
  M6-3 Bloch phase: pattern translated by dx -> field translates (checked on
       Fourier orders with the analytic phase relation), < 1e-6;
  M6-4 (0,0) reflection vs TMM |dr| < 1e-4, converging with p.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lithofem import config, driver, incident

pytestmark = [
    pytest.mark.full,
    pytest.mark.gpu_ok,
    pytest.mark.skipif(not driver.SOLVER_BIN.exists(),
                       reason="lithofem_solve not built"),
]


def _multilayer_config(wl: float, theta: float, pol, diff_layer: bool) -> dict:
    """EUV/DUV-ish stack; optionally one layer expressed as a full-cell frustum."""
    scale = wl / 13.5
    lx = 10.0 * scale
    d_abs, d_gap = 30.0 * scale, 20.0 * scale
    z_diff0, z_diff1 = d_abs, d_abs + 12.0 * scale
    z_max = z_diff1 + d_gap
    c = {
        "domain": {"Lx": lx, "Ly": lx, "z_min": 0.0, "z_max": z_max},
        "materials": {
            "absorber": {"n": 0.95, "k": 0.031},
            "diff": {"epsilon": [1.9, 0.15]},
        },
        "layers": [{"z": [0.0, d_abs], "material": "absorber"}],
        "frustums": [],
        "wavelength": wl,
        "sources": [{
            "type": "planewave",
            "incidence": {"theta": theta, "phi": 0, "from": "top"},
            "polarization": pol,
        }],
        "output": {"planes": [{"z": z_diff1 + 0.6 * d_gap,
                               "quantities": ["E"],
                               "resolution": [16, 16], "file": "obs.h5"}]},
        "fem": {"order": 3, "elems_per_wavelength": 3},
        "boundaries": {"pml": {"thickness": 0.75}},
    }
    if diff_layer:
        c["frustums"] = [{
            "vertices": [[0.0, 0.0], [lx, 0.0], [lx, lx], [0.0, lx]],
            "z0": z_diff0, "h": z_diff1 - z_diff0, "alpha": 90,
            "epsilon": "diff",
        }]
    else:
        # same physical stack, expressed purely as background layers
        c["layers"].append({"z": [z_diff0, z_diff1], "material": "diff"})
    return c


def _tmm_reference_field(model: config.Model, extra_layer: tuple | None):
    """Full-stack TMM background (with the difference layer as a real layer)."""
    raw_layers = [
        {"z": [la.z0, la.z1],
         "material": {"epsilon": [la.eps.real, la.eps.imag]}}
        for la in model.layers
    ]
    if extra_layer is not None:
        z0, z1, eps = extra_layer
        raw_layers.append({"z": [z0, z1], "material": {"epsilon": [eps.real, eps.imag]}})
    src = model.sources[0]
    c = {
        "domain": {"Lx": model.domain.lx, "Ly": model.domain.ly,
                   "z_min": model.domain.z_min, "z_max": model.domain.z_max},
        "layers": raw_layers,
        "frustums": [],
        "wavelength": model.wavelength,
        "sources": [{
            "type": "planewave",
            "incidence": {"theta": src.theta, "phi": src.phi,
                          "from": "top" if src.from_top else "bottom"},
            "polarization": src.polarization,
        }],
        "output": {"planes": [{"z": model.output.planes[0].z,
                               "quantities": ["E"],
                               "resolution": list(model.output.planes[0].resolution),
                               "file": "obs.h5"}]},
    }
    ref_model = config.expand(c)
    return incident.group_incident(ref_model, ref_model.groups[0])


def _solve(c: dict, tmp: Path, order: int | None = None) -> tuple:
    if order is not None:
        c = dict(c)
        c["fem"] = dict(c["fem"], order=order)
    model = config.expand(c)
    prep = driver.prepare(model, tmp)
    meta = driver.solve_group(prep, 0)
    return model, prep, meta


# Always on CPU: an explicit end-to-end anchor for the CPU solver path,
# which is also the automatic fallback path and so must stay verified in
# its own right, not only through the GPU/CPU equivalence tests.
@pytest.mark.cpu_reference
@pytest.mark.parametrize("wl", [13.5, 193.0])
def test_m6_1_pure_multilayer_zero_scattering(wl: float, tmp_path: Path) -> None:
    c = _multilayer_config(wl, 6.0, "s", diff_layer=False)
    model, prep, meta = _solve(c, tmp_path)
    # scattered energy density integral vs incident (|E_inc| ~ 1) x volume
    dom = model.domain
    vol = dom.lx * dom.ly * (dom.z_max - dom.z_min)
    sc = sum(meta["region_l2sq"])
    assert sc / vol < 1e-8, sc / vol


@pytest.mark.parametrize("pol", ["s", "p"])
@pytest.mark.parametrize("theta", [6.0, 17.0])
@pytest.mark.parametrize("wl", [13.5, 193.0])
def test_m6_2_difference_layer_vs_tmm(wl, theta, pol, tmp_path: Path) -> None:
    c = _multilayer_config(wl, theta, pol, diff_layer=True)
    # p=3 with fine z-resolution (the lateral direction is trivial here);
    # measured convergence: epw 4/6/8/10 -> 7.2e-4/2.3e-4/1.1e-4/6.6e-5
    c["fem"] = {"order": 3, "elems_per_wavelength": 10}
    model, prep, meta = _solve(c, tmp_path)
    e_tot, _ = driver.total_field_on_plane(prep, 0, 0)

    fr = model.frustums[0]
    ref = _tmm_reference_field(model, (fr.geom.z_lo, fr.geom.z_hi, fr.eps))
    z = model.output.planes[0].z
    e_ref_hat = ref.eval(np.array([z]))[0]
    x, yv = driver.plane_grid(model, 0)
    kx, ky = model.groups[0].kpar
    phase = np.exp(1j * (ky * yv[:, None] + kx * x[None, :]))
    e_ref = e_ref_hat[None, None, :] * phase[..., None]

    rel = np.linalg.norm(e_tot - e_ref) / np.linalg.norm(e_ref)
    assert rel < 1e-4, f"wl={wl} theta={theta} pol={pol}: rel {rel:.2e}"


def test_m6_2_p_convergence(tmp_path: Path) -> None:
    c = _multilayer_config(13.5, 6.0, "s", diff_layer=True)
    errs = []
    for p in (2, 3, 4):
        model, prep, meta = _solve(c, tmp_path / f"p{p}", order=p)
        e_tot, _ = driver.total_field_on_plane(prep, 0, 0)
        fr = model.frustums[0]
        ref = _tmm_reference_field(model, (fr.geom.z_lo, fr.geom.z_hi, fr.eps))
        z = model.output.planes[0].z
        x, yv = driver.plane_grid(model, 0)
        kx, ky = model.groups[0].kpar
        phase = np.exp(1j * (ky * yv[:, None] + kx * x[None, :]))
        e_ref = ref.eval(np.array([z]))[0][None, None, :] * phase[..., None]
        errs.append(float(np.linalg.norm(e_tot - e_ref) / np.linalg.norm(e_ref)))
    assert errs[0] > errs[1] > errs[2], errs


def test_m6_4_zero_order_reflection_vs_tmm(tmp_path: Path) -> None:
    """(0,0) scattered amplitude + background r vs full-stack TMM r."""
    c = _multilayer_config(13.5, 6.0, "s", diff_layer=True)
    diffs = []
    for p in (2, 3):
        model, prep, meta = _solve(c, tmp_path / f"p{p}", order=p)
        u = driver.load_plane_envelope(prep, 0, 0)
        s00 = complex(u.mean(axis=(0, 1))[1])  # Ey component, s-pol
        # reference the up-going scattered wave to z_max (r convention point)
        from lithofem.constants import k0 as vk0

        k0 = vk0(model.wavelength)
        kx, ky = model.groups[0].kpar
        qz = np.sqrt(k0**2 - kx**2 - ky**2)
        z = model.output.planes[0].z
        c_up = s00 * np.exp(-1j * qz * (z - model.domain.z_max))

        bg = incident.group_incident(model, model.groups[0])
        fr = model.frustums[0]
        full = _tmm_reference_field(model, (fr.geom.z_lo, fr.geom.z_hi, fr.eps))
        # both r_amp are Hy/Ey-basis amplitudes of the reflected wave at z_max;
        # for s-pol the Ey projection of e_s is cos-free (e_s ~ y at phi=0)
        r_num = bg.r_amp + c_up
        diffs.append(abs(r_num - full.r_amp))
    assert diffs[0] > diffs[1], diffs
    assert diffs[1] < 1e-4, diffs


def _pattern_config(shift_x: float = 0.0, order: int = 2, epw: float = 2.0) -> dict:
    """Square-hole EUV pattern (coarse: the M6-3 shift check is exact for any
    discretization, so cost is kept minimal)."""
    lx = 32.0
    v = [[8.0 + shift_x, 8.0], [24.0 + shift_x, 8.0],
         [24.0 + shift_x, 24.0], [8.0 + shift_x, 24.0]]
    return {
        "domain": {"Lx": lx, "Ly": lx, "z_min": 0.0, "z_max": 36.0},
        "materials": {"absorber": {"n": 0.95, "k": 0.031}},
        "layers": [{"z": [0.0, 20.0], "material": "absorber"}],
        "frustums": [{"vertices": v, "z0": 0.0, "h": 20.0, "alpha": 90}],
        "wavelength": 13.5,
        "sources": [{"type": "planewave",
                     "incidence": {"theta": 6, "phi": 0, "from": "top"},
                     "polarization": "s"}],
        "output": {"planes": [{"z": 30.0, "quantities": ["E"],
                               "resolution": [32, 32], "file": "obs.h5"}]},
        "fem": {"order": order, "elems_per_wavelength": epw},
        "boundaries": {"pml": {"thickness": 0.75}},
    }


def test_m6_3_bloch_translation_phase(tmp_path: Path) -> None:
    """Pattern shifted by Lx/2 (same mesh, torus translation): the sampled
    envelope must equal np.roll by nx/2 and order coefficients gain exactly
    (-1)^m -- discretization-independent, checked to 1e-6."""
    lx = 32.0
    model, prep, _ = _solve(_pattern_config(0.0), tmp_path)
    u1 = driver.load_plane_envelope(prep, 0, 0)
    meta2 = driver.solve_group(prep, 0, shift_x=lx / 2)
    u2 = driver.load_plane_envelope(prep, 0, 0)
    # compare Ey: tangential to both x- and z-faces, hence single-valued at
    # sample points that land exactly on element faces (Ex/Ez normal parts
    # jump by the discretization there and depend on which element FindPoints
    # picks -- not an error, see report)
    v1, v2 = u1[..., 1], u2[..., 1]
    nx = v1.shape[1]
    rolled = np.roll(v1, nx // 2, axis=1)
    rel = np.linalg.norm(v2 - rolled) / np.linalg.norm(v1)
    assert rel < 1e-6, rel
    # order relation c'_m = (-1)^m c_m on the strongest orders
    ca = np.fft.fft2(v1) / v1.size
    cb = np.fft.fft2(v2) / v2.size
    ref = np.abs(ca).max()
    checked = 0
    for j in (-1, 0, 1):
        for m in (-2, -1, 0, 1, 2):
            if abs(ca[j, m]) < 1e-3 * ref:
                continue
            rel = abs(cb[j, m] - ca[j, m] * (-1.0) ** m) / ref
            assert rel < 1e-6, (j, m, rel)
            checked += 1
    assert checked >= 5
