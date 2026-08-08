"""Deterministic geometry cases for M3 mesh acceptance (fixed seed).

>= 10 random/structured geometries: concave polygons, alpha != 90, negative
h, frustums spanning multiple layers, periodic wrap, corner refinement.
Domains are small relative to the wavelength so test meshes stay tiny.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .gen_config_cases import _star_polygon

SEED = 20260806


def _base() -> dict[str, Any]:
    return {
        "domain": {"Lx": 60, "Ly": 60, "z_min": 0, "z_max": 40},
        "materials": {"absorber": {"n": 0.95, "k": 0.031}},
        "layers": [{"z": [0, 25], "material": "absorber"}],
        "frustums": [],
        "wavelength": 193.0,
        "sources": [{"type": "planewave", "incidence": {"theta": 6, "phi": 0, "from": "top"}}],
        "fem": {"elems_per_wavelength": 8},
    }


def mesh_cases() -> list[dict[str, Any]]:
    rng = np.random.default_rng(SEED)
    cases: list[dict[str, Any]] = []

    # 1: square hole, alpha=85
    c = _base()
    c["frustums"] = [{"vertices": [[15, 15], [45, 15], [45, 45], [15, 45]],
                      "z0": 0, "h": 25, "alpha": 85}]
    cases.append(c)

    # 2: concave L-shape, alpha=80, spans both layers' interface
    c = _base()
    c["layers"] = [{"z": [0, 15], "material": "absorber"},
                   {"z": [15, 25], "material": "absorber"}]
    c["frustums"] = [{"vertices": [[12, 12], [46, 12], [46, 26], [30, 26],
                                   [30, 46], [12, 46]], "z0": 0, "h": 22, "alpha": 80}]
    cases.append(c)

    # 3: negative h, alpha=95 (expanding away from base)
    c = _base()
    c["frustums"] = [{"vertices": [[20, 20], [40, 20], [40, 40], [20, 40]],
                      "z0": 30, "h": -20, "alpha": 95}]
    cases.append(c)

    # 4: concave star wrapping over +x
    c = _base()
    verts = _star_polygon(rng, 58, 30, 8, 14, 7)
    c["frustums"] = [{"vertices": verts, "z0": 0, "h": 18, "alpha": 88}]
    cases.append(c)

    # 5: expanding star (alpha=100)
    c = _base()
    verts = _star_polygon(rng, 30, 30, 8, 13, 6)
    c["frustums"] = [{"vertices": verts, "z0": 0, "h": 20, "alpha": 100}]
    cases.append(c)

    # 6: two stacked frustums, different alpha
    c = _base()
    c["frustums"] = [
        {"vertices": [[18, 18], [42, 18], [42, 42], [18, 42]], "z0": 0, "h": 12,
         "alpha": 85},
        {"vertices": [[18, 18], [42, 18], [42, 42], [18, 42]], "z0": 12, "h": 12,
         "alpha": 95},
    ]
    cases.append(c)

    # 7: corner-crossing wrap (x and y overhang)
    c = _base()
    c["frustums"] = [{"vertices": [[50, 50], [72, 50], [72, 72], [50, 72]],
                      "z0": 0, "h": 15, "alpha": 87}]
    cases.append(c)

    # 8: vertical prism (alpha=90), concave star
    c = _base()
    verts = _star_polygon(rng, 30, 30, 9, 16, 8)
    c["frustums"] = [{"vertices": verts, "z0": 5, "h": 30}]
    cases.append(c)

    # 9: thin bar, alpha=75 (h within collapse limit)
    c = _base()
    c["frustums"] = [{"vertices": [[10, 27], [50, 27], [50, 33], [10, 33]],
                      "z0": 0, "h": 10, "alpha": 75}]
    cases.append(c)

    # 10: pentagon spanning 3 slabs (both layers + uncovered region)
    c = _base()
    c["layers"] = [{"z": [0, 12], "material": "absorber"},
                   {"z": [12, 25], "material": "absorber"}]
    c["frustums"] = [{"vertices": [[30, 14], [44, 24], [39, 40], [21, 40], [16, 24]],
                      "z0": 0, "h": 32, "alpha": 92}]
    cases.append(c)

    return cases


def corner_refine_case() -> dict[str, Any]:
    c = _base()
    c["frustums"] = [{"vertices": [[20, 20], [40, 20], [40, 40], [20, 40]],
                      "z0": 0, "h": 25}]
    c["fem"] = {"elems_per_wavelength": 8,
                "corner_refine": {"radius": 8.0, "factor": 3.0}}
    return c
