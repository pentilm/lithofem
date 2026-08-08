"""M3 acceptance tests: mesh generation.

Criteria (docs/validation.md):
  M3-1 volume conservation: mesh tet volume sums per material region vs
       analytic frustum/slab volumes, rel < 1e-8, for >= 10 geometries;
  M3-2 conformity: no hanging nodes, all Jacobians > 0;
  M3-3 periodic pairing: x/y boundary nodes pair to < 1e-10;
  M3-4 MFEM loads every mesh and reports matching per-attribute volumes
       (via solver/bin/hello_mfem, built in M0);
  M3-5 corner refinement: element sizes near frustum corner edges smaller
       by >= the configured factor (statistical check).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from lithofem import config, meshcheck, meshgen

from .data_gen import gen_mesh_cases

CASES = gen_mesh_cases.mesh_cases()
HELLO = Path(__file__).resolve().parent.parent / "solver" / "bin" / "hello_mfem"


@pytest.fixture(scope="session")
def meshes(tmp_path_factory: pytest.TempPathFactory) -> list[dict]:
    """Generate all M3 meshes once; return per-case dict of paths/stats."""
    out = []
    root = tmp_path_factory.mktemp("m3_meshes")
    for i, case in enumerate(CASES):
        model = config.expand(case)
        path = root / f"case_{i}.msh"
        info = meshgen.generate(model, path)
        stats = meshcheck.load_stats(str(path))
        out.append({"model": model, "info": info, "stats": stats, "path": path})
    return out


@pytest.mark.full
def test_case_count() -> None:
    assert len(CASES) >= 10


@pytest.mark.full
@pytest.mark.parametrize("i", range(len(CASES)))
def test_volume_conservation(meshes: list[dict], i: int) -> None:
    m = meshes[i]
    analytic = meshgen.analytic_region_volumes(m["model"])
    mesh_vols = m["stats"].region_volumes
    for attr, v_ana in analytic.items():
        v_mesh = mesh_vols.get(attr, 0.0)
        assert v_ana > 0, f"region {attr} has nonpositive analytic volume"
        rel = abs(v_mesh - v_ana) / v_ana
        assert rel < 1e-8, f"case {i} region {attr}: rel error {rel:.2e}"


@pytest.mark.full
@pytest.mark.parametrize("i", range(len(CASES)))
def test_conformity_and_jacobians(meshes: list[dict], i: int) -> None:
    stats = meshes[i]["stats"]
    assert stats.n_hanging_faces == 0
    assert stats.min_jacobian > 0.0


@pytest.mark.full
@pytest.mark.parametrize("i", range(len(CASES)))
def test_periodic_pairing(meshes: list[dict], i: int) -> None:
    m = meshes[i]
    dom = m["model"].domain
    err = meshcheck.periodic_pairing_error(m["stats"], dom.lx, dom.ly)
    assert err < 1e-10, f"case {i}: pairing error {err:.2e}"


@pytest.mark.full
@pytest.mark.parametrize("i", range(len(CASES)))
def test_mfem_loads_and_volumes_agree(meshes: list[dict], i: int) -> None:
    if not HELLO.exists():
        pytest.skip("solver/bin/hello_mfem not built (run `make solver`)")
    m = meshes[i]
    res = subprocess.run(
        [str(HELLO), str(meshgen.mfem_mesh_path(m["path"]))],
        capture_output=True, text=True, timeout=300,
    )
    assert res.returncode == 0, res.stderr
    mfem_vols: dict[int, float] = {}
    for line in res.stdout.splitlines():
        if line.startswith("attribute "):
            parts = line.split()
            mfem_vols[int(parts[1])] = float(parts[3])
    py_vols = m["stats"].region_volumes
    assert set(mfem_vols) == set(py_vols)
    for attr, v in py_vols.items():
        assert abs(mfem_vols[attr] - v) / v < 1e-10, f"attr {attr}"


@pytest.mark.full
def test_corner_refinement(tmp_path: Path) -> None:
    case = gen_mesh_cases.corner_refine_case()
    model = config.expand(case)
    path = tmp_path / "corner.msh"
    meshgen.generate(model, path)
    stats = meshcheck.load_stats(str(path))

    # edge lengths of tets, classified by distance to the frustum corner lines
    f = model.frustums[0].geom
    from lithofem.geometry import mitre_offset_vertices

    lo = mitre_offset_vertices(f.base, f.offset_at(f.z_lo))
    corners = np.array([[x, y, 0.5 * (f.z_lo + f.z_hi)] for x, y in lo])

    xyz = stats.nodes
    tets = stats.tets
    edges = np.concatenate([
        tets[:, [0, 1]], tets[:, [0, 2]], tets[:, [0, 3]],
        tets[:, [1, 2]], tets[:, [1, 3]], tets[:, [2, 3]],
    ])
    mid = 0.5 * (xyz[edges[:, 0]] + xyz[edges[:, 1]])
    ln = np.linalg.norm(xyz[edges[:, 0]] - xyz[edges[:, 1]], axis=1)
    # distance to nearest corner axis (xy distance to corner point, any z)
    d = np.min(
        np.linalg.norm(mid[:, None, :2] - corners[None, :, :2], axis=2), axis=1
    )
    z_ok = (mid[:, 2] > f.z_lo - 2) & (mid[:, 2] < f.z_hi + 2)
    near = ln[(d < 4.0) & z_ok]
    far = ln[d > 20.0]
    assert len(near) > 10 and len(far) > 10
    ratio = np.median(far) / np.median(near)
    factor = model.fem.corner_refine_factor or 0.0
    assert ratio >= factor * 0.9, f"refinement ratio {ratio:.2f} < {factor}"
