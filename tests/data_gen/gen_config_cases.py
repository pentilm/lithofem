"""Deterministic config-case generators for M2 (fixed seed, no external data).

Legal set: >= 20 configs covering concave polygons, negative h, frustums
spanning multiple layers, periodic wrap. Illegal sets: self-intersecting
polygons, offset collapse, frustum overlap, layer overlap (>= 3 each).
"""

from __future__ import annotations

from typing import Any

import numpy as np

SEED = 20260804


def _star_polygon(
    rng: np.random.Generator, cx: float, cy: float, r_lo: float, r_hi: float, n: int
) -> list[list[float]]:
    """Random star-shaped (simple, generally concave) polygon around (cx, cy).

    Angles come from normalized cumulative gaps in [0.5, 1.5], so every angular
    gap is well below pi — this guarantees the radial polygon is simple.
    """
    gaps = rng.uniform(0.5, 1.5, n)
    ang = 2 * np.pi * np.cumsum(gaps) / np.sum(gaps)
    rad = rng.uniform(r_lo, r_hi, n)
    pts = np.stack([cx + rad * np.cos(ang), cy + rad * np.sin(ang)], axis=1)
    return [[float(x), float(y)] for x, y in pts]


def _base(wavelength: float = 13.5) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "domain": {"Lx": 96, "Ly": 96, "z_min": 0, "z_max": 120},
        "materials": {
            "absorber": {"n": 0.95, "k": 0.031},
            "oxide": {"epsilon": [2.13, 0.0]},
        },
        "layers": [
            {"z": [0, 60], "material": "absorber"},
            {"z": [60, 70], "material": "oxide"},
        ],
        "frustums": [],
        "wavelength": wavelength,
        "sources": [
            {"type": "planewave", "incidence": {"theta": 6, "phi": 0, "from": "top"},
             "polarization": "s"},
        ],
    }


def legal_configs() -> list[dict[str, Any]]:
    rng = np.random.default_rng(SEED)
    cases: list[dict[str, Any]] = []

    # 1: minimal five-section config (defaults exercise, M2-4)
    cases.append({
        "domain": {"Lx": 96, "Ly": 96, "z_min": 0, "z_max": 120},
        "materials": {"absorber": {"n": 0.95, "k": 0.031}},
        "layers": [{"z": [0, 60], "material": "absorber"}],
        "frustums": [
            {"vertices": [[24, 24], [72, 24], [72, 72], [24, 72]], "z0": 0, "h": 60},
        ],
        "wavelength": 13.5,
        "sources": [{"type": "planewave", "incidence": {"theta": 6, "phi": 0, "from": "top"}}],
    })

    # 2: square hole with slope (design example)
    c = _base()
    c["frustums"] = [{"vertices": [[24, 24], [72, 24], [72, 72], [24, 72]],
                      "z0": 0, "h": 60, "alpha": 85, "epsilon": 1.0}]
    cases.append(c)

    # 3: negative h
    c = _base()
    c["frustums"] = [{"vertices": [[30, 30], [66, 30], [66, 66], [30, 66]],
                      "z0": 60, "h": -60, "alpha": 85}]
    cases.append(c)

    # 4: frustum spanning multiple layers
    c = _base()
    c["frustums"] = [{"vertices": [[30, 30], [66, 30], [66, 66], [30, 66]],
                      "z0": 0, "h": 70, "alpha": 88}]
    cases.append(c)

    # 5: L-shaped concave polygon
    c = _base()
    c["frustums"] = [{"vertices": [[20, 20], [70, 20], [70, 45], [45, 45],
                                   [45, 70], [20, 70]], "z0": 0, "h": 60, "alpha": 85}]
    cases.append(c)

    # 6: polygon overhanging +x boundary (wrap)
    c = _base()
    c["frustums"] = [{"vertices": [[80, 30], [110, 30], [110, 60], [80, 60]],
                      "z0": 0, "h": 60}]
    cases.append(c)

    # 7: corner-overhanging polygon (wraps in x and y)
    c = _base()
    c["frustums"] = [{"vertices": [[85, 85], [115, 85], [115, 115], [85, 115]],
                      "z0": 0, "h": 40, "alpha": 87}]
    cases.append(c)

    # 8: two disjoint frustums, different materials
    c = _base()
    c["frustums"] = [
        {"vertices": [[10, 10], [40, 10], [40, 40], [10, 40]], "z0": 0, "h": 60,
         "alpha": 85},
        {"vertices": [[55, 55], [85, 55], [85, 85], [55, 85]], "z0": 0, "h": 30,
         "epsilon": "oxide"},
    ]
    cases.append(c)

    # 9: stacked in z (same footprint, disjoint z ranges)
    c = _base()
    c["frustums"] = [
        {"vertices": [[30, 30], [66, 30], [66, 66], [30, 66]], "z0": 0, "h": 30},
        {"vertices": [[30, 30], [66, 30], [66, 66], [30, 66]], "z0": 30, "h": 30,
         "alpha": 95},
    ]
    cases.append(c)

    # 10: alpha > 90 (expanding), negative h
    c = _base()
    c["frustums"] = [{"vertices": [[35, 35], [61, 35], [61, 61], [35, 61]],
                      "z0": 70, "h": -40, "alpha": 100}]
    cases.append(c)

    # 11-20: randomized star polygons (concave), random slopes/signs
    for k in range(10):
        c = _base(wavelength=float(rng.choice([13.5, 193.0])))
        n = int(rng.integers(5, 11))
        verts = _star_polygon(rng, 48, 48, 12, 30, n)
        h = float(rng.uniform(20, 60)) * (1 if k % 2 == 0 else -1)
        z0 = 0.0 if h > 0 else 60.0
        alpha = float(rng.uniform(80, 100))
        if abs(alpha - 90) < 2:
            alpha = 90.0
        if alpha != 90.0:
            # keep the case legal: clamp |h| within the collapse limit
            from lithofem import geometry as _g

            hmax = _g.max_legal_h(_g.polygon_from_vertices(verts), alpha, abs(h))
            if hmax < abs(h):
                h = float(np.sign(h)) * 0.8 * hmax
                z0 = 0.0 if h > 0 else 60.0
        c["frustums"] = [{"vertices": verts, "z0": z0, "h": h, "alpha": alpha}]
        c["sources"] = [{"type": "planewave",
                         "incidence": {"theta": float(rng.uniform(0, 20)),
                                       "phi": float(rng.uniform(0, 360)), "from": "top"},
                         "polarization": "p" if k % 2 else "s"}]
        cases.append(c)

    # 21: multiple sources incl. local ones + output planes
    c = _base()
    c["frustums"] = [{"vertices": [[24, 24], [72, 24], [72, 72], [24, 72]],
                      "z0": 0, "h": 60, "alpha": 85}]
    c["sources"] = [
        {"type": "planewave", "incidence": {"theta": 6, "phi": 0, "from": "top"},
         "polarization": "s"},
        {"type": "point", "position": [48, 48, 90], "current": [[1, 0], [0, 0], [0, 0]]},
        {"type": "sheet", "corner": [0, 0, 100], "edges": [[96, 0, 0], [0, 96, 0]],
         "current": [[1, 0], [0, 0], [0, 0]], "phase_gradient": [0.0, 0.0]},
    ]
    c["output"] = {"planes": [{"z": 61.0, "quantities": ["E", "H"],
                               "resolution": [128, 128], "file": "nf.h5"}]}
    cases.append(c)

    # 22: two plane waves with different k|| (driver split)
    c = _base()
    c["sources"] = [
        {"type": "planewave", "incidence": {"theta": 6, "phi": 0, "from": "top"}},
        {"type": "planewave", "incidence": {"theta": 17, "phi": 90, "from": "top"}},
    ]
    cases.append(c)

    return cases


def illegal_configs() -> dict[str, list[dict[str, Any]]]:
    """Illegal cases keyed by category; each must be rejected with a located error."""
    bad: dict[str, list[dict[str, Any]]] = {
        "self_intersecting": [], "collapse": [], "frustum_overlap": [], "layer_overlap": [],
    }

    # self-intersecting polygons (bow-ties etc.)
    for verts in (
        [[10, 10], [60, 60], [60, 10], [10, 60]],                       # bow-tie
        [[20, 20], [80, 20], [20, 50], [80, 50]],                       # crossing quad
        [[10, 10], [50, 10], [30, 40], [50, 40], [10, 40], [30, 25]],   # crossing hexagon
    ):
        c = _base()
        c["frustums"] = [{"vertices": verts, "z0": 0, "h": 40}]
        bad["self_intersecting"].append(c)

    # offset collapse: shrinking cross-section vanishes before |h| is reached
    collapse_specs = [
        # (vertices, z0, h, alpha): inward rate = |cot(alpha)| per nm of height
        ([[40, 40], [56, 40], [56, 56], [40, 56]], 0, 60, 80),   # 16 nm square, needs > 8 nm offset
        ([[30, 44], [66, 44], [66, 52], [30, 52]], 0, 50, 75),   # thin 8 nm bar
        ([[20, 20], [70, 20], [70, 32], [45, 32], [45, 70], [33, 70], [33, 32],
          [20, 32]], 0, 55, 70),                                  # thin U-shape
    ]
    for verts, z0, h, alpha in collapse_specs:
        c = _base()
        c["frustums"] = [{"vertices": verts, "z0": z0, "h": h, "alpha": alpha}]
        bad["collapse"].append(c)

    # frustum overlap
    overlap_specs = [
        # direct footprint overlap, same z range
        [{"vertices": [[20, 20], [60, 20], [60, 60], [20, 60]], "z0": 0, "h": 50},
         {"vertices": [[50, 50], [90, 50], [90, 90], [50, 90]], "z0": 0, "h": 50}],
        # overlap only via periodic wrap
        [{"vertices": [[80, 30], [110, 30], [110, 60], [80, 60]], "z0": 0, "h": 50},
         {"vertices": [[5, 30], [20, 30], [20, 60], [5, 60]], "z0": 0, "h": 50}],
        # overlap only at upper z (expanding walls meet)
        [{"vertices": [[20, 40], [44, 40], [44, 56], [20, 56]], "z0": 0, "h": 60,
          "alpha": 70},
         {"vertices": [[52, 40], [76, 40], [76, 56], [52, 56]], "z0": 0, "h": 60,
          "alpha": 70}],
    ]
    for fr in overlap_specs:
        c = _base()
        c["frustums"] = fr
        bad["frustum_overlap"].append(c)

    # layer overlap
    for pairs in (
        [[0, 60], [50, 70]],
        [[0, 60], [0, 60], [60, 70]],
        [[10, 40], [39.5, 55]],
    ):
        c = _base()
        c["layers"] = [{"z": z, "material": "absorber"} for z in pairs]
        bad["layer_overlap"].append(c)

    return bad
