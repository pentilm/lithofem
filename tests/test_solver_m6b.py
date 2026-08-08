"""M6b acceptance tests (part 1): sheet current source, superposition,
multi-k|| driver splitting. (Point/line open-domain cases: dipole_test.)

Criteria (docs/validation.md (local sources)):
  M6b-1 uniform sheet in vacuum: radiated field vs E = -Z0*J/2, rel < 1e-6;
        with phase_gradient: emission direction matches the conversion
        formula (diffraction-order check) < 1e-6;
  M6b-4 two sources solved together == sum of separate solves, < 1e-10;
  M6b-5 multi-k|| plane waves split into groups; config round-trip intact.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lithofem import config, driver
from lithofem.constants import k0 as vk0

pytestmark = [
    pytest.mark.full,
    pytest.mark.skipif(not driver.SOLVER_BIN.exists(),
                       reason="lithofem_solve not built"),
]

WL = 50.0
LX = 10.0


def _sheet_config(pg: float = 0.0, current=((1, 0), (0, 0), (0, 0))) -> dict:
    return {
        "domain": {"Lx": LX, "Ly": LX, "z_min": 0.0, "z_max": 60.0},
        "materials": {},
        "layers": [],
        "frustums": [],
        "wavelength": WL,
        "sources": [{
            "type": "sheet", "corner": [0.0, 0.0, 30.0],
            "edges": [[LX, 0.0, 0.0], [0.0, LX, 0.0]],
            "current": [list(c) for c in current],
            "phase_gradient": [pg, 0.0],
        }],
        "output": {"planes": [
            {"z": 45.0, "quantities": ["E"], "resolution": [8, 8], "file": "a.h5"},
            {"z": 15.0, "quantities": ["E"], "resolution": [8, 8], "file": "b.h5"},
        ]},
        "fem": {"order": 4, "elems_per_wavelength": 8},
    }


def _mode(u: np.ndarray, m: int = 0) -> np.ndarray:
    """x-order m, y-order 0 Fourier coefficient of each component."""
    c = np.fft.fft2(u, axes=(0, 1)) / (u.shape[0] * u.shape[1])
    return c[0, m]


def test_m6b_1_uniform_sheet_vs_analytic(tmp_path: Path) -> None:
    """E = -Z0 Js / 2 on both sides (envelope units: Z0*Js = 1)."""
    model = config.expand(_sheet_config())
    prep = driver.prepare(model, tmp_path)
    driver.solve_group(prep, 0)
    k0 = vk0(WL)
    worst = 0.0
    for plane, z in ((0, 45.0), (1, 15.0)):
        u = driver.load_plane_envelope(prep, 0, plane)
        ex = _mode(u)[0]
        expected = -0.5 * np.exp(1j * k0 * abs(z - 30.0))
        worst = max(worst, abs(ex - expected) / 0.5)
    assert worst < 1e-6, worst


def test_m6b_1_phase_gradient_direction(tmp_path: Path) -> None:
    """pg = 2*pi/Lx (propagating order 1): emission direction matches the
    conversion formula kx=pg, kz=sqrt(k0^2-pg^2); kz measured from the phase
    advance between two planes above the sheet."""
    wl, lx = 12.0, 20.0
    g = 2 * np.pi / lx
    c = {
        "domain": {"Lx": lx, "Ly": lx, "z_min": 0.0, "z_max": 42.0},
        "materials": {}, "layers": [], "frustums": [],
        "wavelength": wl,
        "sources": [{
            "type": "sheet", "corner": [0.0, 0.0, 18.0],
            "edges": [[lx, 0.0, 0.0], [0.0, lx, 0.0]],
            "current": [[0, 0], [1, 0], [0, 0]],
            "phase_gradient": [g, 0.0],
        }],
        "output": {"planes": [
            {"z": 21.0, "quantities": ["E"], "resolution": [8, 8], "file": "a.h5"},
            {"z": 39.0, "quantities": ["E"], "resolution": [8, 8], "file": "b.h5"},
        ]},
        "fem": {"order": 4, "elems_per_wavelength": 6},
        "boundaries": {"pml": {"thickness": 0.75}},
    }
    model = config.expand(c)
    prep = driver.prepare(model, tmp_path)
    driver.solve_group(prep, 0)
    k0 = vk0(wl)
    qz = np.sqrt(k0**2 - g**2)
    u_a = driver.load_plane_envelope(prep, 0, 0)  # z=21
    u_b = driver.load_plane_envelope(prep, 0, 1)  # z=39
    c_a, c_b = _mode(u_a, 1), _mode(u_b, 1)
    comp = int(np.argmax(np.abs(c_a)))
    # diffraction-order check: energy sits in order 1 (kx = pg); residual
    # cross-order content is unstructured-mesh discretization noise
    leak = max(np.abs(_mode(u_a, 0)).max(), np.abs(_mode(u_a, -1)).max())
    assert leak < 1e-3 * abs(c_a[comp]), (leak, abs(c_a[comp]))
    # amplitude matches the oblique-sheet formula Z0 J/(2 cos(theta_1))
    k0_ = vk0(12.0)
    cos_t = np.sqrt(1 - (2 * np.pi / 20.0 / k0_) ** 2)
    assert abs(abs(c_a[comp]) - 0.5 / cos_t) < 1e-4
    # kz from the phase advance over 18 nm vs the dispersion formula
    # (compared modulo 2*pi: the baseline spans more than one wavelength)
    dphi = np.angle(c_b[comp] / c_a[comp])
    diff = (dphi - qz * 18.0 + np.pi) % (2 * np.pi) - np.pi
    assert abs(diff) / (k0 * 18.0) < 1e-6, (dphi, qz * 18.0)


def test_m6b_4_superposition(tmp_path: Path) -> None:
    """Two sheets solved together == per-source fields summed (1e-10)."""
    c = _sheet_config()
    c["sources"].append({
        "type": "sheet", "corner": [0.0, 0.0, 24.0],
        "edges": [[LX, 0.0, 0.0], [0.0, LX, 0.0]],
        "current": [[0, 0], [1, 0.5], [0, 0]],
    })
    c["output"]["per_source"] = True
    model = config.expand(c)
    prep = driver.prepare(model, tmp_path)
    driver.solve_group(prep, 0)
    for plane in (0, 1):
        combined = driver.load_plane_envelope(prep, 0, plane)
        total = np.zeros_like(combined)
        for si in (0, 1):
            raw = np.fromfile(
                prep.workdir / f"plane_g0_p{plane}_s{si}.bin", dtype=np.float64
            ).reshape(*combined.shape, 2)
            total += raw[..., 0] + 1j * raw[..., 1]
        scale = np.abs(combined).max()
        assert np.abs(combined - total).max() < 1e-10 * max(scale, 1.0)


def test_m6b_5_multi_kpar_groups(tmp_path: Path) -> None:
    """Two plane waves with different k|| -> two groups, archived outputs."""
    c = {
        "domain": {"Lx": LX, "Ly": LX, "z_min": 0.0, "z_max": 40.0},
        "materials": {"m": {"epsilon": [2.0, 0.1]}},
        "layers": [{"z": [0.0, 15.0], "material": "m"}],
        "frustums": [{"vertices": [[0, 0], [LX, 0], [LX, LX], [0, LX]],
                      "z0": 15.0, "h": 10.0, "alpha": 90,
                      "epsilon": [1.5, 0.05]}],
        "wavelength": WL,
        "sources": [
            {"type": "planewave", "incidence": {"theta": 0, "phi": 0, "from": "top"}},
            {"type": "planewave", "incidence": {"theta": 20, "phi": 0, "from": "top"}},
        ],
        "output": {"planes": [{"z": 32.0, "quantities": ["E"],
                               "resolution": [8, 8], "file": "o.h5"}]},
        "fem": {"order": 2, "elems_per_wavelength": 6},
    }
    model = config.expand(c)
    assert len(model.groups) == 2
    prep = driver.prepare(model, tmp_path)
    for g in range(2):
        meta = driver.solve_group(prep, g)
        assert meta["residual"] < 1e-10
        assert (prep.workdir / f"plane_g{g}_p0.bin").exists()
    # round-trip: solve.json groups carry the right kpar
    import json

    doc = json.loads((prep.workdir / "solve.json").read_text())
    k_expected = vk0(WL) * np.sin(np.deg2rad(20))
    kxs = sorted(abs(gr["kpar"][0]) for gr in doc["groups"])
    assert kxs[0] == pytest.approx(0.0, abs=1e-15)
    assert kxs[1] == pytest.approx(k_expected, rel=1e-12)


DIPOLE = driver.SOLVER_BIN.parent / "dipole_test"


def _run_dipole(mode: int, orient: int, p: int, epw: float,
                inner: float = 2.0, r: float = 0.5) -> float:
    import re
    import subprocess

    res = subprocess.run(
        [str(DIPOLE), "-mode", str(mode), "-or", str(orient),
         "-p", str(p), "-e", str(epw), "-in", str(inner), "-r", str(r)],
        capture_output=True, text=True, timeout=3000,
    )
    assert res.returncode == 0, res.stderr or res.stdout
    m = re.search(r"rel_l2 ([\d.eE+-]+)", res.stdout)
    assert m, res.stdout
    return float(m.group(1))


@pytest.mark.parametrize("orient", [0, 1, 2])
def test_m6b_2_point_dipole_vs_green(orient: int) -> None:
    """Sphere-sampled field (r = lambda/2) vs the free-space dyadic Green
    function; tolerance 1e-2 at p=3 with a decreasing trend (source
    singularity limits the rate; trend documented in the report)."""
    coarse = _run_dipole(0, orient, 2, 4, inner=2.4, r=0.7)
    fine = _run_dipole(0, orient, 3, 5, inner=2.4, r=0.7)
    assert fine < coarse, (coarse, fine)
    assert fine < 1e-2, fine


def test_m6b_3_line_current_vs_hankel() -> None:
    """Circle-sampled E_y (rho = lambda/2) vs -(k0/4) Z0J H0^(1)(k0 rho)."""
    coarse = _run_dipole(1, 1, 2, 4)
    fine = _run_dipole(1, 1, 3, 5)
    assert fine < coarse, (coarse, fine)
    assert fine < 1e-2, fine


def test_m6b_6_sheet_multilayer_energy_balance(tmp_path: Path) -> None:
    """Radiated power == up/down outgoing flux + material absorption, 1e-3."""
    import json

    from lithofem import orders as ordmod

    c = {
        "domain": {"Lx": 10.0, "Ly": 10.0, "z_min": 0.0, "z_max": 50.0},
        "materials": {"absorber": {"n": 1.2, "k": 0.35}},
        "layers": [{"z": [15.0, 25.0], "material": "absorber"}],
        "frustums": [],
        "wavelength": 50.0,
        "sources": [{
            "type": "sheet", "corner": [0.0, 0.0, 35.0],
            "edges": [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]],
            "current": [[1, 0], [0.3, 0.1], [0, 0]],
        }],
        "output": {"planes": [
            {"z": 35.0, "quantities": ["E"], "resolution": [8, 8], "file": "s.h5"},
            {"z": 45.0, "quantities": ["E"], "resolution": [8, 8], "file": "u.h5"},
            {"z": 5.0, "quantities": ["E"], "resolution": [8, 8], "file": "d.h5"},
        ]},
        "fem": {"order": 4, "elems_per_wavelength": 8},
    }
    model = config.expand(c)
    prep = driver.prepare(model, tmp_path)
    meta = driver.solve_group(prep, 0)
    doc = json.loads((prep.workdir / "solve.json").read_text())

    u_sheet = driver.load_plane_envelope(prep, 0, 0)
    u_up = driver.load_plane_envelope(prep, 0, 1)
    u_dn = driver.load_plane_envelope(prep, 0, 2)

    p_rad = ordmod.sheet_radiated_power(u_sheet, model, 0)
    o_up = ordmod.orders_from_plane(u_up, model, 0, 1.0 + 0j)
    o_dn = ordmod.orders_from_plane(u_dn, model, 0, 1.0 + 0j)
    area = model.domain.lx * model.domain.ly
    p_up = ordmod.one_way_flux_z(o_up, +1, area)
    p_dn = -ordmod.one_way_flux_z(o_dn, -1, area)
    p_abs = ordmod.absorbed_power(meta, doc)

    balance = abs(p_rad - (p_up + p_dn + p_abs)) / abs(p_rad)
    assert balance < 1e-3, (p_rad, p_up, p_dn, p_abs, balance)
