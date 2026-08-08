"""Ten deliberately broken YAML configs (M10 destructive testing)."""

from __future__ import annotations

BAD_CONFIGS: dict[str, str] = {
    "unknown_field": """
domain: {Lx: 10, Ly: 10, z_min: 0, z_max: 20}
wavelenght: 13.5
sources: [{type: planewave}]
""",
    "bad_type_domain": """
domain: {Lx: ten, Ly: 10, z_min: 0, z_max: 20}
wavelength: 13.5
sources: [{type: planewave}]
""",
    "inverted_z": """
domain: {Lx: 10, Ly: 10, z_min: 20, z_max: 0}
wavelength: 13.5
sources: [{type: planewave}]
""",
    "negative_wavelength": """
domain: {Lx: 10, Ly: 10, z_min: 0, z_max: 20}
wavelength: -5
sources: [{type: planewave}]
""",
    "unknown_material": """
domain: {Lx: 10, Ly: 10, z_min: 0, z_max: 20}
layers: [{z: [0, 10], material: unobtainium}]
wavelength: 13.5
sources: [{type: planewave}]
""",
    "layer_overlap": """
domain: {Lx: 10, Ly: 10, z_min: 0, z_max: 20}
materials: {m: {epsilon: [2, 0]}}
layers: [{z: [0, 12], material: m}, {z: [8, 20], material: m}]
wavelength: 13.5
sources: [{type: planewave}]
""",
    "self_intersecting_polygon": """
domain: {Lx: 10, Ly: 10, z_min: 0, z_max: 20}
frustums: [{vertices: [[1,1],[9,9],[9,1],[1,9]], z0: 0, h: 10}]
wavelength: 13.5
sources: [{type: planewave}]
""",
    "bad_polarization": """
domain: {Lx: 10, Ly: 10, z_min: 0, z_max: 20}
wavelength: 13.5
sources: [{type: planewave, polarization: circular}]
""",
    "negative_k": """
domain: {Lx: 10, Ly: 10, z_min: 0, z_max: 20}
materials: {gain: {n: 1.5, k: -0.2}}
layers: [{z: [0, 10], material: gain}]
wavelength: 13.5
sources: [{type: planewave}]
""",
    "alpha_out_of_range": """
domain: {Lx: 10, Ly: 10, z_min: 0, z_max: 20}
frustums: [{vertices: [[2,2],[8,2],[8,8],[2,8]], z0: 0, h: 10, alpha: 200}]
wavelength: 13.5
sources: [{type: planewave}]
""",
}
