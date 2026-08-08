"""M2 acceptance tests: YAML schema, validation, polygon/frustum geometry,
slab computation, solve.json generation.

Criteria (docs/validation.md):
  M2-1 legal set (>= 20, concave/negative-h/multi-layer/wrap) passes; slab
       partitions match 5 hand-derived benchmarks;
  M2-2 illegal sets (self-intersection, offset collapse, frustum overlap,
       layer overlap; >= 3 each) rejected with object index + suggestion;
       collapse errors report max |h| within 1% of a bisection reference;
  M2-3 wrap conserves total area to 1e-10 (relative);
  M2-4 minimal five-section config expands to a complete legal solve.json.
"""

from __future__ import annotations

import json
import re

import numpy as np
import pytest

from lithofem import config, geometry

from .data_gen import gen_config_cases as gen

LEGAL = gen.legal_configs()
ILLEGAL = gen.illegal_configs()


@pytest.mark.fast
def test_legal_case_count() -> None:
    assert len(LEGAL) >= 20


@pytest.mark.fast
@pytest.mark.parametrize("i", range(len(LEGAL)))
def test_legal_configs_expand(i: int) -> None:
    model = config.expand(LEGAL[i])
    doc = config.model_to_solve_json(model)
    json.dumps(doc)  # must be serializable
    assert doc["wavelength"] > 0
    assert len(doc["regions"]) >= len(model.slabs) - 1
    assert len(model.groups) >= 1


# --- M2-1: hand-derived slab benchmarks -----------------------------------

SLAB_BENCHMARKS = [
    # (legal case index, expected slab boundaries)
    (1, [0, 60, 70, 120]),        # design example: layer bounds only (frustum ends coincide)
    (2, [0, 60, 70, 120]),        # negative h frustum 60 -> 0
    (3, [0, 60, 70, 120]),        # frustum spans 0..70, ends coincide with layer bounds
    (8, [0, 30, 60, 70, 120]),    # stacked frustums add breakpoint at 30
    (20, [0, 60, 61, 70, 100, 120]),  # obs plane z=61 + sheet source z=100
]


@pytest.mark.fast
@pytest.mark.parametrize("case_idx,expected", SLAB_BENCHMARKS)
def test_slab_benchmarks(case_idx: int, expected: list[float]) -> None:
    model = config.expand(LEGAL[case_idx])
    assert np.allclose(model.slabs, expected, atol=1e-12), (model.slabs, expected)


@pytest.mark.fast
def test_eps_bg_per_slab() -> None:
    model = config.expand(LEGAL[1])
    eps_absorber = config.epsilon_from_nk(0.95, 0.031)
    assert model.eps_bg_of_slab(0) == pytest.approx(eps_absorber)
    assert model.eps_bg_of_slab(1) == pytest.approx(2.13 + 0j)
    assert model.eps_bg_of_slab(2) == pytest.approx(1.0 + 0j)


# --- M2-2: illegal inputs --------------------------------------------------


@pytest.mark.fast
@pytest.mark.parametrize("category", sorted(ILLEGAL))
def test_illegal_category_counts(category: str) -> None:
    assert len(ILLEGAL[category]) >= 3


@pytest.mark.fast
@pytest.mark.parametrize(
    "category,j",
    [(c, j) for c in sorted(ILLEGAL) for j in range(len(ILLEGAL[c]))],
)
def test_illegal_configs_rejected(category: str, j: int) -> None:
    with pytest.raises(config.ConfigError) as exc:
        config.expand(ILLEGAL[category][j])
    msg = str(exc.value)
    where = "layers" if category == "layer_overlap" else "frustum"
    assert where in msg
    assert re.search(r"frustums?\[\d+\]", msg) or category == "layer_overlap"
    assert "—" in msg  # every issue carries a fix suggestion


@pytest.mark.fast
@pytest.mark.parametrize("j", range(3))
def test_collapse_reports_max_h_within_1pct(j: int) -> None:
    case = ILLEGAL["collapse"][j]
    fr = case["frustums"][0]
    with pytest.raises(config.ConfigError) as exc:
        config.expand(case)
    m = re.search(r"maximum allowed \|h\| is ([0-9.eE+-]+)", str(exc.value))
    assert m, f"no max-|h| report in: {exc.value}"
    reported = float(m.group(1))
    base = geometry.polygon_from_vertices(fr["vertices"])
    truth = geometry.max_legal_h(base, fr["alpha"], abs(fr["h"]))
    assert truth < abs(fr["h"])  # sanity: case is genuinely illegal
    assert abs(reported - truth) <= 0.01 * truth, (reported, truth)


# --- M2-3: wrap area conservation -----------------------------------------


@pytest.mark.fast
def test_wrap_area_conservation() -> None:
    rng = np.random.default_rng(20260805)
    lx = ly = 96.0
    worst = 0.0
    for _ in range(20):
        cx = rng.uniform(-20, 116)
        cy = rng.uniform(-20, 116)
        n = int(rng.integers(4, 10))
        poly = geometry.polygon_from_vertices(
            gen._star_polygon(rng, cx, cy, 8, 25, n)
        )
        pieces = geometry.wrap_polygon(poly, lx, ly)
        total = sum(p.area for p in pieces)
        worst = max(worst, abs(total - poly.area) / poly.area)
    assert worst < 1e-10, worst


@pytest.mark.fast
def test_wrap_case_expands() -> None:
    model = config.expand(LEGAL[5])  # overhanging +x
    base = model.frustums[0].geom.base
    pieces = geometry.wrap_polygon(base, model.domain.lx, model.domain.ly)
    assert len(pieces) == 2
    assert abs(sum(p.area for p in pieces) - base.area) / base.area < 1e-10


# --- M2-4: minimal config full expansion ----------------------------------


@pytest.mark.fast
def test_minimal_config_defaults() -> None:
    model = config.expand(LEGAL[0])
    doc = config.model_to_solve_json(model)
    assert doc["fem"]["order"] == 3
    assert doc["fem"]["elems_per_wavelength"] == 4.0
    assert doc["solver"]["type"] == "direct"
    assert doc["solver"]["device"] == "cpu"
    assert doc["pml"]["thickness_wavelengths"] == 1.0
    assert doc["pml"]["order"] == 2
    assert doc["pml"]["target_reflection"] == 1e-8
    assert doc["lateral_bc"] == "periodic"
    assert doc["output"]["orders"]["enabled"] is True
    assert doc["conventions"]["time"] == "exp(-i*omega*t)"
    assert doc["conventions"]["length_unit"] == "nm"
    # plane wave got its k|| computed
    src = doc["sources"][0]
    k_expected = 2 * np.pi / 13.5 * np.sin(np.deg2rad(6))
    assert src["kpar"][0] == pytest.approx(k_expected, rel=1e-12)
    assert src["kpar"][1] == pytest.approx(0.0, abs=1e-15)


# --- misc semantics --------------------------------------------------------


@pytest.mark.fast
def test_two_planewaves_split_into_groups() -> None:
    model = config.expand(LEGAL[21])
    assert len(model.groups) == 2


@pytest.mark.fast
def test_local_sources_join_planewave_group() -> None:
    model = config.expand(LEGAL[20])
    (group,) = [g for g in model.groups]
    assert set(group.source_indices) == {0, 1, 2}


@pytest.mark.fast
def test_ccw_normalization() -> None:
    cw = [[0, 0], [0, 10], [10, 10], [10, 0]]  # clockwise input
    poly = geometry.polygon_from_vertices(cw)
    assert poly.exterior.is_ccw


@pytest.mark.fast
def test_only_local_sources_uses_bloch_k() -> None:
    c = gen._base()
    c["sources"] = [{"type": "point", "position": [48, 48, 90],
                     "current": [[1, 0], [0, 0], [0, 0]]}]
    c["bloch_k"] = [0.01, 0.0]
    model = config.expand(c)
    assert model.groups[0].kpar == (0.01, 0.0)
