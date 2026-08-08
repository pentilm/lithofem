"""Thin Python wrapper over the YAML/CLI pipeline (docs/physics.md, M10).

    import lithofem
    lithofem.run("config.yaml", outdir="results/")

plus dataclass-style config builders that only *emit YAML* — they contain no
logic of their own (the YAML/CLI path is the single source of truth).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .cli import run as _run


def run(config_path: str | Path, outdir: str | Path = "lithofem_out",
        device: str | None = None) -> Path:
    """Identical to `lithofem run config.yaml -o outdir` (same code path)."""
    return _run(config_path, outdir, device=device)


# --------------------------------------------------------------------------
# dataclass config builders (convenience for parameter sweeps)
# --------------------------------------------------------------------------


@dataclass
class Frustum:
    vertices: list[list[float]]
    z0: float
    h: float
    alpha: float = 90.0
    epsilon: Any = 1.0


@dataclass
class Layer:
    z: list[float]
    material: str


@dataclass
class PlaneWave:
    theta: float = 0.0
    phi: float = 0.0
    from_: str = "top"
    polarization: Any = "s"
    amplitude: Any = 1.0

    def to_dict(self) -> dict:
        return {"type": "planewave", "amplitude": self.amplitude,
                "incidence": {"theta": self.theta, "phi": self.phi,
                              "from": self.from_},
                "polarization": self.polarization}


@dataclass
class OutputPlane:
    z: float
    quantities: list[str] = field(default_factory=lambda: ["E"])
    resolution: list[int] = field(default_factory=lambda: [256, 256])
    file: str = "plane.h5"


@dataclass
class Config:
    """Programmatic config; `save()` writes schema-conformant YAML."""

    lx: float
    ly: float
    z_min: float
    z_max: float
    wavelength: float
    materials: dict[str, Any] = field(default_factory=dict)
    layers: list[Layer] = field(default_factory=list)
    frustums: list[Frustum] = field(default_factory=list)
    sources: list[Any] = field(default_factory=list)
    planes: list[OutputPlane] = field(default_factory=list)
    fem_order: int = 3
    elems_per_wavelength: float = 4.0

    def to_dict(self) -> dict:
        doc: dict[str, Any] = {
            "schema_version": 1,
            "domain": {"Lx": self.lx, "Ly": self.ly,
                       "z_min": self.z_min, "z_max": self.z_max},
            "materials": self.materials,
            "layers": [{"z": la.z, "material": la.material}
                       for la in self.layers],
            "frustums": [{"vertices": f.vertices, "z0": f.z0, "h": f.h,
                          "alpha": f.alpha, "epsilon": f.epsilon}
                         for f in self.frustums],
            "wavelength": self.wavelength,
            "sources": [s.to_dict() if hasattr(s, "to_dict") else s
                        for s in self.sources],
            "fem": {"order": self.fem_order,
                    "elems_per_wavelength": self.elems_per_wavelength},
        }
        if self.planes:
            doc["output"] = {"planes": [
                {"z": p.z, "quantities": p.quantities,
                 "resolution": p.resolution, "file": p.file}
                for p in self.planes]}
        return doc

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)
        return path
