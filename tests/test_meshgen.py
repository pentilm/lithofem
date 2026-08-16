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


# --- v2.6: 3D meshing algorithm -----------------------------------------

@pytest.mark.full
def test_hxt_matches_delaunay(tmp_path: Path) -> None:
    """The optional HXT path (fem.mesh_algorithm: hxt) must produce a mesh
    geometrically equivalent to the default Delaunay one: the same region
    volumes to round-off, comparable element counts, and positive Jacobians
    throughout.

    Element-for-element identity is *not* expected — these are different
    tetrahedralizations of the same domain at the same target size. Nor is
    equal *accuracy*: at equal target size the ~10% sparser HXT mesh measurably
    degrade the hardest acceptance case (M8-1 TM vs RCWA), which is why
    Delaunay is the default and HXT is opt-in for meshing-bound geometries.
    """
    import os

    model = config.expand(CASES[0])
    results = {}
    for name, algo in (("hxt", "10"), ("delaunay", "1")):
        os.environ["LITHOFEM_MESH_ALGO3D"] = algo
        try:
            path = tmp_path / f"{name}.msh"
            info = meshgen.generate(model, path)
            results[name] = (info, meshcheck.load_stats(str(path)))
        finally:
            os.environ.pop("LITHOFEM_MESH_ALGO3D", None)

    (info_h, st_h), (info_d, st_d) = results["hxt"], results["delaunay"]

    # same geometry: per-region volumes agree to round-off
    assert set(info_h.region_volumes) == set(info_d.region_volumes)
    for attr, vol_h in info_h.region_volumes.items():
        vol_d = info_d.region_volumes[attr]
        assert abs(vol_h - vol_d) <= 1e-9 * abs(vol_d), (attr, vol_h, vol_d)

    # same resolution: element counts within 25% (different tetrahedralizations
    # of the same size field, not the same mesh)
    n_h, n_d = st_h.n_tets, st_d.n_tets
    assert 0.75 < n_h / n_d < 1.33, (n_h, n_d)

    # HXT output is a valid mesh in its own right
    assert st_h.min_jacobian > 0.0, st_h.min_jacobian
    assert st_h.n_hanging_faces == 0, st_h.n_hanging_faces
