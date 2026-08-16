"""M7 acceptance tests: outputs (HDF5 slices, ParaView, diffraction orders,
energy conservation).

Criteria (docs/validation.md):
  M7-1 pure multilayer: all orders except (0,0) < 1e-10 in energy;
       (0,0) vs TMM |dr| < 1e-6;
  M7-2 lossless grating: sum of order fluxes vs incident flux < 1e-4;
  M7-3 HDF5 independently readable, complete metadata, slice z on a mesh
       face (z within 1e-10 of a slab breakpoint);
  M7-4 ParaView file loads in pyvista; 5 sampled points agree with the
       solver's plane samples < 1e-8.
"""

from __future__ import annotations

import json

import h5py
import numpy as np
import pytest

from lithofem import config, driver, incident, orders, outputs
from lithofem.constants import k0 as vk0

pytestmark = [
    pytest.mark.full,
    pytest.mark.gpu_ok,
    pytest.mark.skipif(not driver.SOLVER_BIN.exists(),
                       reason="lithofem_solve not built"),
]

WL = 50.0
LX = 10.0


def _stack_config(with_grating: bool, volume: bool = False,
                  epw: float = 8.0, order: int = 3) -> dict:
    ly = 4.0 if with_grating else LX  # grating is y-invariant: thin cell
    c = {
        "domain": {"Lx": LX, "Ly": ly, "z_min": 0.0, "z_max": 50.0},
        "materials": {"hi": {"epsilon": [2.25, 0.0]},
                      "lo": {"epsilon": [1.5, 0.0]}},
        "layers": [{"z": [0.0, 15.0], "material": "hi"}],
        "frustums": [],
        "wavelength": WL,
        "sources": [{"type": "planewave",
                     "incidence": {"theta": 6, "phi": 0, "from": "top"},
                     "polarization": "s"}],
        "output": {
            "planes": [
                {"z": 42.0, "quantities": ["E", "H"],
                 "resolution": [16, 16], "file": "top.h5"},
                {"z": 4.0, "quantities": ["E"],
                 "resolution": [16, 16], "file": "bot.h5"},
            ],
            "volume": {"enabled": volume, "file": "vol", "include_pml": False},
        },
        "fem": {"order": order, "elems_per_wavelength": epw},
    }
    if with_grating:
        # lossless lamellar grating: half-cell bar of "lo" inside vacuum gap
        c["frustums"] = [{
            "vertices": [[0.0, 0.0], [LX / 2, 0.0], [LX / 2, ly], [0.0, ly]],
            "z0": 20.0, "h": 10.0, "alpha": 90, "epsilon": "lo",
        }]
    return c


@pytest.fixture(scope="module")
def multilayer_run(tmp_path_factory: pytest.TempPathFactory):
    model = config.expand(_stack_config(False))
    prep = driver.prepare(model, tmp_path_factory.mktemp("m7_ml"))
    meta = driver.solve_group(prep, 0)
    return model, prep, meta


@pytest.fixture(scope="module")
def grating_run(tmp_path_factory: pytest.TempPathFactory):
    model = config.expand(_stack_config(True, volume=True, epw=12.0))
    prep = driver.prepare(model, tmp_path_factory.mktemp("m7_gr"))
    meta = driver.solve_group(prep, 0)
    return model, prep, meta


@pytest.fixture(scope="module")
def grating_run_coarse(tmp_path_factory: pytest.TempPathFactory):
    # the lateral size cap (Lx/4) fixes h here, so refine via p instead
    model = config.expand(_stack_config(True, epw=12.0, order=2))
    prep = driver.prepare(model, tmp_path_factory.mktemp("m7_grc"))
    meta = driver.solve_group(prep, 0)
    return model, prep, meta


def test_m7_1_multilayer_orders(multilayer_run) -> None:
    model, prep, meta = multilayer_run
    h5p, csvp = outputs.write_orders_files(prep, 0, 0, 1.0 + 0j, +1)
    with h5py.File(h5p) as f:
        m = f["m"][:]
        n = f["n"][:]
        flux = f["flux_z"][:]
        amp_s = f["amp_s_re"][:] + 1j * f["amp_s_im"][:]
    inc = incident.group_incident(model, model.groups[0])
    kx, ky = model.groups[0].kpar
    k0 = vk0(WL)
    qz = np.sqrt(k0**2 - kx**2 - ky**2)
    inc_flux = 0.5 * (qz / k0) * model.domain.lx * model.domain.ly
    others = (np.abs(flux) / inc_flux)[(m != 0) | (n != 0)]
    assert others.max() < 1e-10, others.max()
    r00 = complex(amp_s[(m == 0) & (n == 0)][0])
    assert abs(r00 - inc.r_amp) < 1e-6, (r00, inc.r_amp)


def _grating_balance(run) -> float:
    model, prep, meta = run
    doc = json.loads((prep.workdir / "solve.json").read_text())
    area = model.domain.lx * model.domain.ly
    k0 = vk0(WL)
    kx, ky = model.groups[0].kpar
    qz_in = np.sqrt(k0**2 - kx**2 - ky**2)
    inc_flux = 0.5 * (qz_in / k0) * area

    # reflection side (vacuum): total field orders = scattered + analytic (0,0)
    u_top = driver.load_plane_envelope(prep, 0, 0)
    inc_f = incident.group_incident(model, model.groups[0])
    z_top = model.output.planes[0].z
    e_inc = inc_f.eval(np.array([z_top]))[0]
    # split incident into up (reflected) and down (incoming) at this plane:
    # above the stack only the reflected part goes up; subtract the incoming
    # wave and add the full background so the top plane carries E_total.
    u_tot_top = u_top + e_inc[None, None, :]
    o_top = orders.orders_from_plane(u_tot_top, model, 0, 1.0 + 0j)
    # up-flux: remove the incoming part order by order is impossible from one
    # plane; instead compute net flux = up - down through the top plane and
    # use  R_net = incoming - net  ... net flux positive down = absorbed+T.
    # For a lossless stack: incident = R + T  <=>  net_down_top = T.
    # net flux from total field needs E and H: use physical fields.
    fields_top = outputs.plane_fields(prep, 0, 0)
    e = fields_top["E"]
    h = fields_top["H"]
    s_z = 0.5 * np.real(np.cross(e, np.conj(h))[..., 2])
    net_down_top = -s_z.mean() * area  # +z up; net downward power

    fields_bot = outputs.plane_fields(prep, 0, 1)
    e_b = fields_bot["E"]
    h_b = fields_bot["H"]
    s_zb = 0.5 * np.real(np.cross(e_b, np.conj(h_b))[..., 2])
    t_flux = -s_zb.mean() * area  # transmitted (down through bottom plane)

    # lossless: net power entering through the top == power leaving bottom
    balance = abs(net_down_top - t_flux) / inc_flux

    # order-resolved transmission: sum of order fluxes at the bottom == total
    o_bot = orders.orders_from_plane(
        driver.load_plane_envelope(prep, 0, 1)
        + inc_f.eval(np.array([model.output.planes[1].z]))[0][None, None, :],
        model, 0, complex(2.25),
    )
    t_orders = -orders.one_way_flux_z(o_bot, -1, area)
    assert abs(t_orders - t_flux) / inc_flux < 2e-4, (t_orders, t_flux)
    return float(balance)


def test_m7_2_lossless_grating_energy(grating_run, grating_run_coarse) -> None:
    fine = _grating_balance(grating_run)
    coarse = _grating_balance(grating_run_coarse)
    assert fine < 1e-4, fine
    assert fine < coarse, (coarse, fine)  # converges under refinement


def test_m7_3_h5_metadata(multilayer_run) -> None:
    model, prep, meta = multilayer_run
    path = outputs.write_plane_h5(prep, 0, 0)
    with h5py.File(path) as f:
        assert f.attrs["wavelength_nm"] == WL
        assert f.attrs["time_convention"] == "exp(-i*omega*t)"
        assert f.attrs["length_unit"] == "nm"
        assert f.attrs["angle_unit"] == "degree"
        assert f.attrs["mu_r"] == 1.0
        srcs = json.loads(f.attrs["sources_json"])
        assert srcs[0]["type"] == "planewave"
        assert srcs[0]["theta_deg"] == 6
        assert f["E_re"].shape == (16, 16, 3)
        assert f["H_re"].shape == (16, 16, 3)
        z = float(f.attrs["z"])
    assert min(abs(z - s) for s in model.slabs) < 1e-10


def test_m7_4_paraview_crosscheck(grating_run) -> None:
    import pyvista as pv

    model, prep, meta = grating_run
    pvd = prep.workdir / "vol_g0" / "vol_g0.pvd"
    assert pvd.exists(), list(prep.workdir.rglob("*.pvd"))
    grid = pv.read(str(pvd))
    if isinstance(grid, pv.MultiBlock):
        grid = grid.combine()
    u = driver.load_plane_envelope(prep, 0, 0)
    x, y = driver.plane_grid(model, 0)
    zp = model.output.planes[0].z
    pts = []
    refs = []
    for (i, j) in [(2, 3), (5, 7), (8, 11), (12, 4), (14, 14)]:
        pts.append([x[i] + 3.1e-7, y[j] + 1.7e-7, zp - 4.9e-7])
        refs.append(u[j, i])
    cloud = pv.PolyData(np.array(pts))
    sampled = cloud.sample(grid, tolerance=1e-6)
    worst = 0.0
    for k in range(5):
        num = sampled["Esc_env_re"][k] + 1j * sampled["Esc_env_im"][k]
        worst = max(worst, np.abs(num - refs[k]).max())
    assert worst < 1e-8, worst
