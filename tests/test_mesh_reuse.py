"""v2.6: mesh reuse (cache, user-supplied mesh) and the pre-flight probe.

Reuse is a correctness hazard before it is a speed feature — serving a stale
mesh would silently solve the wrong problem. These tests pin the two things
that make it safe: the cache key changes whenever anything the mesher reads
changes, and it does *not* change for settings that leave the mesh alone.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lithofem import config, driver, meshgen

pytestmark = [
    pytest.mark.full,
    pytest.mark.skipif(not driver.SOLVER_BIN.exists(),
                       reason="lithofem_solve not built"),
]


def _cfg(**over) -> dict:
    c = {
        "domain": {"Lx": 24.0, "Ly": 24.0, "z_min": 0.0, "z_max": 40.0},
        "materials": {"a": {"n": 1.5, "k": 0.0}},
        "layers": [{"z": [0.0, 20.0], "material": "a"}],
        "frustums": [{"vertices": [[6.0, 6.0], [18.0, 6.0], [18.0, 18.0],
                                   [6.0, 18.0]],
                      "z0": 0.0, "h": 20.0, "alpha": 90, "epsilon": 1.0}],
        "wavelength": 13.5,
        "sources": [{"type": "planewave",
                     "incidence": {"theta": 0, "phi": 0, "from": "top"},
                     "polarization": "s"}],
        "fem": {"order": 1, "elems_per_wavelength": 2.0},
        "output": {"planes": [{"z": 30.0, "quantities": ["E"],
                               "resolution": [8, 8], "file": "o.h5"}]},
    }
    c.update(over)
    return c


def _key(**over) -> str:
    return driver.mesh_key(config.expand(_cfg(**over)))


# --- cache key ----------------------------------------------------------

def test_key_ignores_settings_that_do_not_change_the_mesh() -> None:
    """FE order, sources and solver settings leave the mesh untouched, so a
    p-refinement study or an angle sweep must reuse one mesh."""
    base = _key()
    assert _key(fem={"order": 3, "elems_per_wavelength": 2.0}) == base
    assert _key(solver={"type": "direct", "device": "gpu"}) == base
    assert _key(sources=[{"type": "planewave",
                          "incidence": {"theta": 12, "phi": 0, "from": "top"},
                          "polarization": "p"}]) == base


@pytest.mark.parametrize("field,value", [
    ("domain", {"Lx": 26.0, "Ly": 24.0, "z_min": 0.0, "z_max": 40.0}),
    ("wavelength", 20.0),
    ("fem", {"order": 1, "elems_per_wavelength": 3.0}),
    ("materials", {"a": {"n": 2.5, "k": 0.0}}),
    ("layers", [{"z": [0.0, 15.0], "material": "a"}]),
    ("frustums", [{"vertices": [[6.0, 6.0], [16.0, 6.0], [16.0, 16.0],
                                [6.0, 16.0]],
                   "z0": 0.0, "h": 20.0, "alpha": 90, "epsilon": 1.0}]),
    ("boundaries", {"pml": {"thickness": 1.2}}),
    ("output", {"planes": [{"z": 31.0, "quantities": ["E"],
                            "resolution": [8, 8], "file": "o.h5"}]}),
])
def test_key_changes_when_the_mesh_would_change(field: str, value) -> None:
    """Every input the mesher reads must invalidate the key, including the
    observation-plane z values (they become mesh breakpoints) and the
    materials (their refractive index sets the element size)."""
    assert _key(**{field: value}) != _key()


# --- reuse paths --------------------------------------------------------

def test_cache_reuse_reproduces_the_mesh(tmp_path: Path) -> None:
    model = config.expand(_cfg())
    cache = tmp_path / "cache"
    first = driver.prepare(model, tmp_path / "a", cache_dir=cache)
    second = driver.prepare(model, tmp_path / "b", cache_dir=cache)
    assert (tmp_path / "a" / "mesh.msh").read_bytes() == \
           (tmp_path / "b" / "mesh.msh").read_bytes()
    assert second.mesh_info.n_regions == first.mesh_info.n_regions
    for attr, vol in first.mesh_info.region_volumes.items():
        assert second.mesh_info.region_volumes[attr] == pytest.approx(
            vol, rel=1e-6)


def test_user_supplied_mesh_is_used_verbatim(tmp_path: Path) -> None:
    model = config.expand(_cfg())
    src = driver.prepare(model, tmp_path / "gen")
    reused = driver.prepare(model, tmp_path / "use", mesh=src.mesh_path)
    assert (tmp_path / "use" / "mesh.msh").read_bytes() == \
           src.mesh_path.read_bytes()
    assert meshgen.mfem_periodic_mesh_path(reused.mesh_path).exists()


def test_user_supplied_mesh_missing_file_is_reported(tmp_path: Path) -> None:
    model = config.expand(_cfg())
    with pytest.raises(FileNotFoundError):
        driver.prepare(model, tmp_path / "x", mesh=tmp_path / "nope.msh")


# --- pre-flight probe ---------------------------------------------------

def test_probe_reports_size_without_solving(tmp_path: Path) -> None:
    model = config.expand(_cfg())
    prep = driver.prepare(model, tmp_path)
    pr = driver.probe(prep)
    assert pr.ndof > 0 and pr.elements > 0
    assert pr.vram_estimate_gb > 0.0
    # the probe stops before assembly, so no solution artefacts appear
    assert not (tmp_path / "solve_meta_g0.json").exists()


def test_vram_estimate_matches_measurements() -> None:
    """The fit must reproduce the measured cuDSS footprints it came from."""
    for ndof, measured in ((52_167, 0.40), (282_762, 4.00),
                           (460_670, 7.90), (1_344_564, 35.0)):
        est = driver.vram_estimate_gb(ndof)
        assert abs(est - measured) / measured < 0.06, (ndof, est, measured)


# --- v2.6: mesh_algorithm / mesh_threads options -------------------------

def test_mesh_options_validate() -> None:
    m = config.expand(_cfg(fem={"order": 1, "elems_per_wavelength": 2.0,
                                "mesh_algorithm": "hxt", "mesh_threads": 4}))
    assert m.fem.mesh_algorithm == "hxt"
    assert m.fem.mesh_threads == 4
    with pytest.raises(config.ConfigError):
        config.expand(_cfg(fem={"order": 1, "elems_per_wavelength": 2.0,
                                "mesh_algorithm": "fast"}))
    with pytest.raises(config.ConfigError):
        config.expand(_cfg(fem={"order": 1, "elems_per_wavelength": 2.0,
                                "mesh_threads": 0}))


def test_key_tracks_mesh_algorithm_and_threads() -> None:
    """A cached mesh must never cross algorithm or determinism boundaries:
    an hxt mesh must not satisfy a delaunay request, and a threaded
    (non-reproducible) mesh must not satisfy a deterministic one."""
    base = _key()
    hxt = _key(fem={"order": 1, "elems_per_wavelength": 2.0,
                    "mesh_algorithm": "hxt"})
    thr = _key(fem={"order": 1, "elems_per_wavelength": 2.0,
                    "mesh_threads": 8})
    assert hxt != base
    assert thr != base
    assert hxt != thr


def test_mesh_algorithm_option_reaches_gmsh(tmp_path: Path) -> None:
    """The config option (not just the env var) must drive the mesher: the
    two algorithms tetrahedralize differently, so the meshes must differ."""
    from lithofem import meshcheck

    m_d = config.expand(_cfg())
    m_h = config.expand(_cfg(fem={"order": 1, "elems_per_wavelength": 2.0,
                                  "mesh_algorithm": "hxt"}))
    p_d = driver.prepare(m_d, tmp_path / "d")
    p_h = driver.prepare(m_h, tmp_path / "h")
    n_d = meshcheck.load_stats(str(p_d.mesh_path)).n_tets
    n_h = meshcheck.load_stats(str(p_h.mesh_path)).n_tets
    assert n_d != n_h, (n_d, n_h)


def test_supplied_mesh_for_periodic_model_requires_per_sibling(
        tmp_path: Path) -> None:
    """A periodic solve consumes the .per.msh sibling; its absence must be
    reported at prepare time, not later inside the solver."""
    model = config.expand(_cfg())
    src = driver.prepare(model, tmp_path / "gen")
    per = meshgen.mfem_periodic_mesh_path(src.mesh_path)
    per.unlink()
    with pytest.raises(FileNotFoundError, match="per"):
        driver.prepare(model, tmp_path / "use", mesh=src.mesh_path)
