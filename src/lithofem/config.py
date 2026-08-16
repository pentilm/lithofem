"""YAML configuration schema, validation and expansion (docs/physics.md, M2).

Errors are collected (not fail-fast) and reported together with the object
index and a fix suggestion (docs/physics.md). The expanded internal model is a
`Model`; `model_to_solve_json` produces the machine-format snapshot consumed
by the C++ solver (stage-A/C file contract).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from . import geometry
from .constants import ANGLE_UNIT, LENGTH_UNIT, TIME_CONVENTION, epsilon_from_nk, k0

SCHEMA_VERSION = 1

_TOP_KEYS = {
    "schema_version", "domain", "lateral_bc", "background_epsilon", "materials",
    "layers", "frustums", "wavelength", "sources", "fem", "solver", "output",
    "boundaries", "bloch_k", "sweep",
}


class ConfigError(ValueError):
    """All validation problems, each with location + suggestion."""

    def __init__(self, issues: list[str]):
        self.issues = issues
        super().__init__("configuration invalid:\n  - " + "\n  - ".join(issues))


# --------------------------------------------------------------------------
# expanded model dataclasses
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Domain:
    lx: float
    ly: float
    z_min: float
    z_max: float


@dataclass(frozen=True)
class Layer:
    z0: float
    z1: float
    eps: complex
    name: str


@dataclass(frozen=True)
class Frustum:
    geom: geometry.FrustumGeometry
    eps: complex
    material: str


@dataclass(frozen=True)
class PlaneWave:
    amplitude: complex
    theta: float
    phi: float
    from_top: bool
    polarization: str  # "s" | "p" | "jones"
    jones: tuple[complex, complex] | None
    kpar: tuple[float, float]  # k0 sin(theta) (cos phi, sin phi), rad/nm


@dataclass(frozen=True)
class PointSource:
    position: tuple[float, float, float]
    current: tuple[complex, complex, complex]


@dataclass(frozen=True)
class LineSource:
    endpoints: tuple[tuple[float, float, float], tuple[float, float, float]]
    current: tuple[complex, complex, complex]
    phase_gradient: float


@dataclass(frozen=True)
class SheetSource:
    corner: tuple[float, float, float]
    edges: tuple[tuple[float, float, float], tuple[float, float, float]]
    current: tuple[complex, complex, complex]
    phase_gradient: tuple[float, float]


Source = PlaneWave | PointSource | LineSource | SheetSource


@dataclass(frozen=True)
class FemParams:
    order: int = 3
    elems_per_wavelength: float = 4.0
    corner_refine_radius: float | None = None
    corner_refine_factor: float | None = None
    assembly: str = "cpu"  # cpu | gpu (v2.5: GPU matrix assembly)
    mesh_threads: int = 1  # >1 meshes faster but is not bit-reproducible
    mesh_algorithm: str = "delaunay"  # delaunay | hxt (faster, coarser; see docs)


@dataclass(frozen=True)
class SolverParams:
    type: str = "direct"
    device: str = "cpu"
    gpu_ids: tuple[int, ...] = (0,)
    method: str = "gmres"
    rtol: float = 1e-8
    max_iter: int = 2000
    gpu_mem_gb: float = 0.0  # v2: VRAM cap for cuDSS (0 = auto: free VRAM)


@dataclass(frozen=True)
class SweepParams:
    """Task-level multi-GPU sweep (v2, docs/gpu.md): solve groups are
    dispatched to a process pool, round-robin over gpu_ids."""
    gpu_ids: tuple[int, ...] = (0,)
    max_parallel: int = 1


@dataclass(frozen=True)
class PmlParams:
    thickness_wavelengths: float = 1.0
    order: int = 2
    target_reflection: float = 1e-8


@dataclass(frozen=True)
class OutputPlane:
    z: float
    quantities: tuple[str, ...]
    resolution: tuple[int, int]
    file: str


@dataclass(frozen=True)
class OutputParams:
    planes: tuple[OutputPlane, ...] = ()
    volume_enabled: bool = False
    volume_file: str = "field_full"
    volume_include_pml: bool = False
    orders_enabled: bool = True
    per_source: bool = False


@dataclass(frozen=True)
class SolveGroup:
    """Sources sharing one Bloch wavevector -> one linear solve."""

    kpar: tuple[float, float]
    source_indices: tuple[int, ...]


@dataclass(frozen=True)
class Model:
    domain: Domain
    lateral_bc: str
    background_eps: complex
    layers: tuple[Layer, ...]
    frustums: tuple[Frustum, ...]
    wavelength: float
    sources: tuple[Source, ...]
    groups: tuple[SolveGroup, ...]
    fem: FemParams
    solver: SolverParams
    pml: PmlParams
    output: OutputParams
    slabs: tuple[float, ...]  # z breakpoints, ascending, includes z_min/z_max
    sweep: SweepParams | None = None  # v2: task-level multi-GPU dispatch

    def eps_bg_of_slab(self, i: int) -> complex:
        """Background permittivity of slab i (between slabs[i], slabs[i+1])."""
        zc = 0.5 * (self.slabs[i] + self.slabs[i + 1])
        for layer in self.layers:
            if layer.z0 <= zc <= layer.z1:
                return layer.eps
        return self.background_eps


# --------------------------------------------------------------------------
# parsing helpers
# --------------------------------------------------------------------------


def _cnum(v: Any) -> complex:
    """Accept float or [re, im] as a complex number."""
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return complex(v)
    if isinstance(v, (list, tuple)) and len(v) == 2:
        return complex(float(v[0]), float(v[1]))
    raise ValueError(f"expected number or [re, im], got {v!r}")


def _cvec3(v: Any) -> tuple[complex, complex, complex]:
    if not isinstance(v, (list, tuple)) or len(v) != 3:
        raise ValueError(f"expected 3 complex components [[re,im]x3], got {v!r}")
    return (_cnum(v[0]), _cnum(v[1]), _cnum(v[2]))


class _Ctx:
    def __init__(self) -> None:
        self.issues: list[str] = []

    def err(self, where: str, msg: str, fix: str) -> None:
        self.issues.append(f"{where}: {msg} — {fix}")


def _resolve_material(
    spec: Any, materials: dict[str, complex], where: str, ctx: _Ctx
) -> complex:
    """Material reference: name, scalar epsilon, or {n,k}/{epsilon} mapping."""
    try:
        if isinstance(spec, str):
            if spec not in materials:
                ctx.err(where, f"unknown material {spec!r}",
                        f"define it under materials: (known: {sorted(materials)})")
                return 1.0 + 0j
            return materials[spec]
        if isinstance(spec, dict):
            return _material_from_dict(spec)
        return _cnum(spec)
    except ValueError as e:
        ctx.err(where, str(e), "use a material name, a number, [re,im], or {n,k}")
        return 1.0 + 0j


def _material_from_dict(spec: dict[str, Any]) -> complex:
    if "epsilon" in spec:
        return _cnum(spec["epsilon"])
    if "n" in spec:
        n = float(spec["n"])
        kk = float(spec.get("k", 0.0))
        if kk < 0:
            raise ValueError(f"k must be >= 0 (lossy convention Im eps > 0), got {kk}")
        return epsilon_from_nk(n, kk)
    raise ValueError(f"material needs 'epsilon' or 'n'/'k', got keys {sorted(spec)}")


# --------------------------------------------------------------------------
# main entry
# --------------------------------------------------------------------------


def load_yaml(path: str | Path) -> Model:
    with open(path) as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ConfigError(["top level: YAML must be a mapping — see docs example"])
    return expand(raw)


def expand(raw: dict[str, Any]) -> Model:  # noqa: PLR0912, PLR0915
    ctx = _Ctx()

    for key in raw:
        if key not in _TOP_KEYS:
            ctx.err(f"'{key}'", "unknown top-level section",
                    f"valid sections: {sorted(_TOP_KEYS)}")

    ver = raw.get("schema_version", SCHEMA_VERSION)
    if ver != SCHEMA_VERSION:
        ctx.err("schema_version", f"unsupported version {ver}", f"use {SCHEMA_VERSION}")

    # --- domain -----------------------------------------------------------
    dom_raw = raw.get("domain")
    domain = Domain(1.0, 1.0, 0.0, 1.0)
    if not isinstance(dom_raw, dict):
        ctx.err("domain", "missing or not a mapping",
                "provide domain: {Lx, Ly, z_min, z_max}")
    else:
        try:
            domain = Domain(
                float(dom_raw["Lx"]), float(dom_raw["Ly"]),
                float(dom_raw["z_min"]), float(dom_raw["z_max"]),
            )
            if domain.lx <= 0 or domain.ly <= 0:
                ctx.err("domain", "Lx and Ly must be positive", "check the cell size")
            if domain.z_max <= domain.z_min:
                ctx.err("domain", "z_max must exceed z_min", "swap or fix the z range")
        except (KeyError, TypeError, ValueError) as e:
            ctx.err("domain", f"bad field ({e})", "need numeric Lx, Ly, z_min, z_max")

    lateral_bc = raw.get("lateral_bc", "periodic")
    if lateral_bc not in ("periodic", "pml"):
        ctx.err("lateral_bc", f"must be 'periodic' or 'pml', got {lateral_bc!r}",
                "fix the value")
    elif lateral_bc == "pml":
        ctx.err("lateral_bc", "'pml' is scheduled for v1.1 and not yet available",
                "use 'periodic'")

    # --- materials --------------------------------------------------------
    materials: dict[str, complex] = {}
    for name, spec in (raw.get("materials") or {}).items():
        try:
            materials[str(name)] = (
                _material_from_dict(spec) if isinstance(spec, dict) else _cnum(spec)
            )
        except (ValueError, TypeError) as e:
            ctx.err(f"materials.{name}", str(e), "use {n: ..., k: ...} or {epsilon: [re, im]}")

    background_eps = _resolve_material(
        raw.get("background_epsilon", 1.0), materials, "background_epsilon", ctx
    )

    # --- layers -----------------------------------------------------------
    layers: list[Layer] = []
    for i, lay in enumerate(raw.get("layers") or []):
        where = f"layers[{i}]"
        if not isinstance(lay, dict) or "z" not in lay or "material" not in lay:
            ctx.err(where, "needs {z: [z0, z1], material: ...}", "fix the entry")
            continue
        try:
            z0, z1 = float(lay["z"][0]), float(lay["z"][1])
        except (TypeError, ValueError, IndexError):
            ctx.err(where, f"bad z interval {lay.get('z')!r}", "use z: [z0, z1]")
            continue
        if z1 <= z0:
            ctx.err(where, f"empty/inverted z interval [{z0}, {z1}]", "ensure z1 > z0")
            continue
        eps = _resolve_material(lay["material"], materials, where + ".material", ctx)
        name = lay["material"] if isinstance(lay["material"], str) else f"layer{i}"
        layers.append(Layer(z0, z1, eps, name))
    layers.sort(key=lambda la: la.z0)
    for a, b in zip(layers, layers[1:], strict=False):
        if b.z0 < a.z1 - 1e-12:
            ctx.err("layers", f"intervals [{a.z0},{a.z1}] and [{b.z0},{b.z1}] overlap",
                    "make layer z ranges disjoint (uncovered gaps use background)")

    # --- frustums ---------------------------------------------------------
    frustums: list[Frustum] = []
    for i, fr in enumerate(raw.get("frustums") or []):
        where = f"frustums[{i}]"
        if not isinstance(fr, dict):
            ctx.err(where, "not a mapping", "see docs example")
            continue
        if "holes" in fr or "edge_alpha" in fr:
            ctx.err(where, "'holes'/'edge_alpha' are reserved for a future version",
                    "remove the field (v1 does not implement it)")
        try:
            alpha = float(fr.get("alpha", 90.0))
            h = float(fr["h"])
            z0 = float(fr["z0"])
        except (KeyError, TypeError, ValueError) as e:
            ctx.err(where, f"bad z0/h/alpha ({e})", "need numeric z0, h (alpha optional)")
            continue
        if not 0.0 < alpha < 180.0:
            ctx.err(where, f"alpha={alpha} out of range", "use 0 < alpha < 180 (degrees)")
            continue
        if h == 0.0:
            ctx.err(where, "h must be nonzero", "give the frustum a height (h<0 goes down)")
            continue
        try:
            base = geometry.polygon_from_vertices(fr["vertices"])
        except (KeyError, geometry.GeometryError) as e:
            ctx.err(where + ".vertices", str(e), "provide a simple polygon")
            continue
        geom = geometry.FrustumGeometry(base, z0, h, alpha, index=i)
        try:
            geometry.validate_frustum_extrusion(geom)
        except geometry.GeometryError as e:
            ctx.err(where, str(e), "see the maximum |h| reported above")
            continue
        eps = _resolve_material(fr.get("epsilon", 1.0), materials, where + ".epsilon", ctx)
        mat = fr.get("epsilon", 1.0)
        frustums.append(
            Frustum(geom, eps, mat if isinstance(mat, str) else f"frustum{i}")
        )

    # frustums must stay within the z-domain
    for f in frustums:
        if f.geom.z_lo < domain.z_min - 1e-9 or f.geom.z_hi > domain.z_max + 1e-9:
            ctx.err(f"frustums[{f.geom.index}]",
                    f"z-range [{f.geom.z_lo}, {f.geom.z_hi}] exceeds the domain",
                    "clip z0/h to [z_min, z_max]")

    # --- wavelength & sources --------------------------------------------
    wavelength = 0.0
    try:
        wavelength = float(raw["wavelength"])
        if wavelength <= 0:
            raise ValueError
    except (KeyError, TypeError, ValueError):
        ctx.err("wavelength", f"missing or invalid ({raw.get('wavelength')!r})",
                "give the vacuum wavelength in nm (e.g. 13.5)")
        wavelength = 1.0

    sources: list[Source] = []
    for i, src in enumerate(raw.get("sources") or []):
        where = f"sources[{i}]"
        parsed = _parse_source(src, where, wavelength, ctx)
        if parsed is not None:
            sources.append(parsed)
    if not sources:
        ctx.err("sources", "no valid sources", "define at least one source")

    # --- fem / solver / pml / output -------------------------------------
    fem = _parse_fem(raw.get("fem") or {}, ctx)
    solver = _parse_solver(raw.get("solver") or {}, ctx)
    sweep = _parse_sweep(raw.get("sweep"), ctx)
    pml = _parse_pml((raw.get("boundaries") or {}).get("pml") or {}, ctx)
    output = _parse_output(raw.get("output") or {}, domain, ctx)

    # --- solve groups (Bloch k uniqueness, docs/configuration.md) ----------------
    groups = _solve_groups(sources, raw.get("bloch_k"), ctx)

    # --- slabs ------------------------------------------------------------
    sheet_zs = [
        s.corner[2] for s in sources
        if isinstance(s, SheetSource)
    ]
    slabs = geometry.z_breakpoints(
        domain.z_min, domain.z_max,
        [z for lay in layers for z in (lay.z0, lay.z1)],
        [z for f in frustums for z in (f.geom.z_lo, f.geom.z_hi)],
        [p.z for p in output.planes] + sheet_zs,
    )

    # --- frustum overlap (uses slab breakpoints) -------------------------
    per = lateral_bc == "periodic"
    for msg in geometry.frustum_overlap_errors(
        [f.geom for f in frustums],
        domain.lx if per else None, domain.ly if per else None, slabs,
    ):
        ctx.err("frustums", msg, "frustums must not overlap")

    if ctx.issues:
        raise ConfigError(ctx.issues)

    return Model(
        domain=domain, lateral_bc=lateral_bc, background_eps=background_eps,
        layers=tuple(layers), frustums=tuple(frustums), wavelength=wavelength,
        sources=tuple(sources), groups=groups, fem=fem, solver=solver, pml=pml,
        output=output, slabs=tuple(slabs), sweep=sweep,
    )


def _parse_source(
    src: Any, where: str, wavelength: float, ctx: _Ctx
) -> Source | None:
    if not isinstance(src, dict) or "type" not in src:
        ctx.err(where, "source needs a 'type'", "one of planewave|point|line|sheet")
        return None
    t = src["type"]
    try:
        if t == "planewave":
            inc = src.get("incidence") or {}
            theta = float(inc.get("theta", 0.0))
            phi = float(inc.get("phi", 0.0))
            frm = inc.get("from", "top")
            if frm not in ("top", "bottom"):
                raise ValueError(f"incidence.from must be top|bottom, got {frm!r}")
            if not 0.0 <= theta < 90.0:
                raise ValueError(f"theta must be in [0, 90) deg, got {theta}")
            pol = src.get("polarization", "s")
            jones: tuple[complex, complex] | None = None
            if isinstance(pol, dict):
                jones = (_cnum(pol["jones"][0]), _cnum(pol["jones"][1]))
                pol = "jones"
            elif pol not in ("s", "p"):
                raise ValueError(f"polarization must be s|p|{{jones: ...}}, got {pol!r}")
            kk = k0(wavelength) * np.sin(np.deg2rad(theta))
            kpar = (kk * float(np.cos(np.deg2rad(phi))), kk * float(np.sin(np.deg2rad(phi))))
            return PlaneWave(
                amplitude=_cnum(src.get("amplitude", 1.0)), theta=theta, phi=phi,
                from_top=(frm == "top"), polarization=pol, jones=jones, kpar=kpar,
            )
        if t == "point":
            pos = src["position"]
            return PointSource(
                position=(float(pos[0]), float(pos[1]), float(pos[2])),
                current=_cvec3(src["current"]),
            )
        if t == "line":
            e = src["endpoints"]
            p0 = (float(e[0][0]), float(e[0][1]), float(e[0][2]))
            p1 = (float(e[1][0]), float(e[1][1]), float(e[1][2]))
            if np.linalg.norm(np.subtract(p1, p0)) < 1e-12:
                raise ValueError("line endpoints coincide")
            return LineSource(
                endpoints=(p0, p1), current=_cvec3(src["current"]),
                phase_gradient=float(src.get("phase_gradient", 0.0)),
            )
        if t == "sheet":
            c = src["corner"]
            e1, e2 = src["edges"]
            v1 = np.array([float(x) for x in e1])
            v2 = np.array([float(x) for x in e2])
            if np.linalg.norm(np.cross(v1, v2)) < 1e-12:
                raise ValueError("sheet edge vectors are parallel/degenerate")
            if abs(v1[2]) > 1e-12 or abs(v2[2]) > 1e-12:
                raise ValueError(
                    "v1 supports horizontal (z-normal) sheets only; "
                    "set both edge z-components to 0"
                )
            pg = src.get("phase_gradient", [0.0, 0.0])
            return SheetSource(
                corner=(float(c[0]), float(c[1]), float(c[2])),
                edges=(tuple(v1.tolist()), tuple(v2.tolist())),
                current=_cvec3(src["current"]),
                phase_gradient=(float(pg[0]), float(pg[1])),
            )
        raise ValueError(f"unknown source type {t!r}")
    except (KeyError, TypeError, ValueError, IndexError) as e:
        ctx.err(where, f"invalid {t!r} source: {e}", "see the config reference")
        return None


def _parse_fem(f: dict[str, Any], ctx: _Ctx) -> FemParams:
    try:
        order = int(f.get("order", 3))
        if order < 1 or order != float(f.get("order", 3)):
            raise ValueError
    except (TypeError, ValueError):
        ctx.err("fem.order", f"must be a positive integer, got {f.get('order')!r}",
                "use e.g. order: 3")
        order = 3
    cr = f.get("corner_refine") or {}
    try:
        mesh_threads = int(f.get("mesh_threads", 1))
        if mesh_threads < 1:
            raise ValueError
    except (TypeError, ValueError):
        ctx.err("fem.mesh_threads",
                f"must be a positive integer, got {f.get('mesh_threads')!r}",
                "use 1 (default, reproducible) or a thread count")
        mesh_threads = 1
    mesh_algorithm = f.get("mesh_algorithm", "delaunay")
    if mesh_algorithm not in ("delaunay", "hxt"):
        ctx.err("fem.mesh_algorithm",
                f"must be delaunay|hxt, got {mesh_algorithm!r}",
                "delaunay is the accuracy-neutral default")
        mesh_algorithm = "delaunay"
    assembly = f.get("assembly", "cpu")
    if assembly not in ("cpu", "gpu"):
        ctx.err("fem.assembly", f"must be cpu|gpu, got {assembly!r}",
                "use assembly: gpu for GPU matrix assembly (v2.5)")
        assembly = "cpu"
    return FemParams(
        order=order,
        elems_per_wavelength=float(f.get("elems_per_wavelength", 4.0)),
        corner_refine_radius=float(cr["radius"]) if "radius" in cr else None,
        corner_refine_factor=float(cr["factor"]) if "factor" in cr else None,
        assembly=assembly,
        mesh_threads=mesh_threads,
        mesh_algorithm=mesh_algorithm,
    )


def _parse_solver(s: dict[str, Any], ctx: _Ctx) -> SolverParams:
    typ = s.get("type", "direct")
    dev = s.get("device", "cpu")
    if typ not in ("direct", "iterative"):
        ctx.err("solver.type", f"must be direct|iterative, got {typ!r}", "fix the value")
        typ = "direct"
    if dev not in ("cpu", "gpu"):
        ctx.err("solver.device", f"must be cpu|gpu, got {dev!r}", "fix the value")
        dev = "cpu"
    gpu_ids = tuple(int(i) for i in s.get("gpu_ids", [0]))
    return SolverParams(
        type=typ, device=dev, gpu_ids=gpu_ids,
        method=s.get("method", "gmres"),
        rtol=float(s.get("rtol", 1e-8)), max_iter=int(s.get("max_iter", 2000)),
        gpu_mem_gb=float(s.get("gpu_mem_gb", 0.0)),
    )


def _parse_sweep(s: dict[str, Any] | None, ctx: _Ctx) -> SweepParams | None:
    if s is None:
        return None
    try:
        gpu_ids = tuple(int(i) for i in s.get("gpu_ids", [0]))
    except (TypeError, ValueError):
        ctx.err("sweep.gpu_ids", "must be a list of integers", "e.g. [0, 1]")
        gpu_ids = (0,)
    if not gpu_ids or any(i < 0 for i in gpu_ids):
        ctx.err("sweep.gpu_ids", "must be non-empty, ids >= 0", "e.g. [0, 1]")
        gpu_ids = (0,)
    max_parallel = int(s.get("max_parallel", len(gpu_ids)))
    if max_parallel < 1:
        ctx.err("sweep.max_parallel", "must be >= 1", "raise the value")
        max_parallel = 1
    unknown = set(s) - {"gpu_ids", "max_parallel"}
    if unknown:
        ctx.err("sweep", f"unknown keys {sorted(unknown)}",
                "valid keys: gpu_ids, max_parallel")
    return SweepParams(gpu_ids=gpu_ids, max_parallel=max_parallel)


def _parse_pml(p: dict[str, Any], ctx: _Ctx) -> PmlParams:
    try:
        return PmlParams(
            thickness_wavelengths=float(p.get("thickness", 1.0)),
            order=int(p.get("order", 2)),
            target_reflection=float(p.get("target_reflection", 1e-8)),
        )
    except (TypeError, ValueError) as e:
        ctx.err("boundaries.pml", f"bad value ({e})",
                "use {thickness: <wavelengths>, order: <int>, target_reflection: <float>}")
        return PmlParams()


def _parse_output(o: dict[str, Any], domain: Domain, ctx: _Ctx) -> OutputParams:
    planes: list[OutputPlane] = []
    for i, p in enumerate(o.get("planes") or []):
        where = f"output.planes[{i}]"
        try:
            z = float(p["z"])
            if not domain.z_min <= z <= domain.z_max:
                ctx.err(where, f"z={z} outside the domain", "choose z inside [z_min, z_max]")
                continue
            res = p.get("resolution", [256, 256])
            q = tuple(p.get("quantities", ["E"]))
            bad = [x for x in q if x not in ("E", "H")]
            if bad:
                ctx.err(where, f"unknown quantities {bad}", "use E and/or H")
                continue
            planes.append(OutputPlane(
                z=z, quantities=q, resolution=(int(res[0]), int(res[1])),
                file=str(p.get("file", f"plane_z{z:g}.h5")),
            ))
        except (KeyError, TypeError, ValueError) as e:
            ctx.err(where, f"bad plane spec ({e})", "need {z, quantities, resolution, file}")
    vol = o.get("volume") or {}
    orders = o.get("orders") or {}
    return OutputParams(
        planes=tuple(planes),
        volume_enabled=bool(vol.get("enabled", False)),
        volume_file=str(vol.get("file", "field_full")),
        volume_include_pml=bool(vol.get("include_pml", False)),
        orders_enabled=bool(orders.get("enabled", True)),
        per_source=bool(o.get("per_source", False)),
    )


def _solve_groups(
    sources: list[Source], bloch_k_raw: Any, ctx: _Ctx
) -> tuple[SolveGroup, ...]:
    """Group sources by Bloch k|| (docs/configuration.md): one k|| per solve.

    Plane waves define their own k||; local sources join every group (they
    inherit that solve's Bloch phasing). With only local sources, k|| is
    bloch_k (default 0) and a one-time hint is implied (logged by the CLI).
    """
    pw = [(i, s) for i, s in enumerate(sources) if isinstance(s, PlaneWave)]
    local = [i for i, s in enumerate(sources) if not isinstance(s, PlaneWave)]
    groups: list[SolveGroup] = []
    if pw:
        if bloch_k_raw is not None:
            ctx.err("bloch_k", "cannot be combined with plane-wave sources",
                    "remove bloch_k (k|| follows from the incidence angles)")
        seen: dict[tuple[float, float], list[int]] = {}
        for i, s in pw:
            key = (round(s.kpar[0], 15), round(s.kpar[1], 15))
            seen.setdefault(key, []).append(i)
        for key, idxs in seen.items():
            groups.append(SolveGroup(kpar=key, source_indices=tuple(idxs + local)))
    else:
        kb = (0.0, 0.0)
        if bloch_k_raw is not None:
            try:
                kb = (float(bloch_k_raw[0]), float(bloch_k_raw[1]))
            except (TypeError, ValueError, IndexError):
                ctx.err("bloch_k", f"bad value {bloch_k_raw!r}", "use [kx, ky] in rad/nm")
        if local:
            groups.append(SolveGroup(kpar=kb, source_indices=tuple(local)))
    return tuple(groups)


# --------------------------------------------------------------------------
# solve.json
# --------------------------------------------------------------------------


def _c(z: complex) -> list[float]:
    return [float(z.real), float(z.imag)]


def model_to_solve_json(model: Model) -> dict[str, Any]:
    """Machine-format snapshot for the C++ solver (file contract, R-SOL-7).

    Region/attribute convention shared with the mesh generator: mesh volume
    attribute = 1-based index into `regions`, enumerated slab-major then
    (background, frustum 0, frustum 1, ...) within each slab; only regions
    that actually contain material in that slab are listed. The two z-PML
    slabs (below z_min / above z_max) come last, in that order.
    """
    regions = []
    for i in range(len(model.slabs) - 1):
        z_lo, z_hi = model.slabs[i], model.slabs[i + 1]
        zc = 0.5 * (z_lo + z_hi)
        regions.append({
            "slab": i, "z": [z_lo, z_hi], "kind": "background",
            "epsilon": _c(model.eps_bg_of_slab(i)),
        })
        for j, f in enumerate(model.frustums):
            if f.geom.z_lo <= zc <= f.geom.z_hi:
                regions.append({
                    "slab": i, "z": [z_lo, z_hi], "kind": "frustum", "frustum": j,
                    "epsilon": _c(f.eps),
                })
    t_pml = model.pml.thickness_wavelengths * model.wavelength
    n_slab = len(model.slabs) - 1
    regions.append({
        "slab": -1, "z": [model.domain.z_min - t_pml, model.domain.z_min],
        "kind": "pml_bottom", "epsilon": _c(model.eps_bg_of_slab(0)),
    })
    regions.append({
        "slab": n_slab, "z": [model.domain.z_max, model.domain.z_max + t_pml],
        "kind": "pml_top", "epsilon": _c(model.eps_bg_of_slab(n_slab - 1)),
    })

    sources = []
    for s in model.sources:
        if isinstance(s, PlaneWave):
            sources.append({
                "type": "planewave", "amplitude": _c(s.amplitude),
                "theta": s.theta, "phi": s.phi,
                "from": "top" if s.from_top else "bottom",
                "polarization": s.polarization,
                "jones": [_c(s.jones[0]), _c(s.jones[1])] if s.jones else None,
                "kpar": list(s.kpar),
            })
        elif isinstance(s, PointSource):
            sources.append({
                "type": "point", "position": list(s.position),
                "current": [_c(c) for c in s.current],
            })
        elif isinstance(s, LineSource):
            sources.append({
                "type": "line", "endpoints": [list(p) for p in s.endpoints],
                "current": [_c(c) for c in s.current],
                "phase_gradient": s.phase_gradient,
            })
        else:
            sources.append({
                "type": "sheet", "corner": list(s.corner),
                "edges": [list(e) for e in s.edges],
                "current": [_c(c) for c in s.current],
                "phase_gradient": list(s.phase_gradient),
            })

    return {
        "schema_version": SCHEMA_VERSION,
        "conventions": {
            "time": TIME_CONVENTION, "length_unit": LENGTH_UNIT,
            "angle_unit": ANGLE_UNIT, "mu_r": 1.0,
        },
        "wavelength": model.wavelength,
        "k0": k0(model.wavelength),
        "domain": {
            "Lx": model.domain.lx, "Ly": model.domain.ly,
            "z_min": model.domain.z_min, "z_max": model.domain.z_max,
        },
        "lateral_bc": model.lateral_bc,
        "background_epsilon": _c(model.background_eps),
        "layers": [
            {"z": [la.z0, la.z1], "epsilon": _c(la.eps), "name": la.name}
            for la in model.layers
        ],
        "slabs": list(model.slabs),
        "regions": regions,
        "sources": sources,
        "groups": [
            {"kpar": list(g.kpar), "source_indices": list(g.source_indices)}
            for g in model.groups
        ],
        "fem": {
            "order": model.fem.order,
            "elems_per_wavelength": model.fem.elems_per_wavelength,
            "assembly": model.fem.assembly,
            "mesh_threads": model.fem.mesh_threads,
            "mesh_algorithm": model.fem.mesh_algorithm,
            "corner_refine": (
                {"radius": model.fem.corner_refine_radius,
                 "factor": model.fem.corner_refine_factor}
                if model.fem.corner_refine_radius is not None else None
            ),
        },
        "solver": {
            "type": model.solver.type, "device": model.solver.device,
            "gpu_ids": list(model.solver.gpu_ids), "method": model.solver.method,
            "rtol": model.solver.rtol, "max_iter": model.solver.max_iter,
            "gpu_mem_gb": model.solver.gpu_mem_gb,
        },
        "pml": {
            "thickness_wavelengths": model.pml.thickness_wavelengths,
            "order": model.pml.order,
            "target_reflection": model.pml.target_reflection,
        },
        "output": {
            "planes": [
                {"z": p.z, "quantities": list(p.quantities),
                 "resolution": list(p.resolution), "file": p.file}
                for p in model.output.planes
            ],
            "volume": {
                "enabled": model.output.volume_enabled,
                "file": model.output.volume_file,
                "include_pml": model.output.volume_include_pml,
            },
            "orders": {"enabled": model.output.orders_enabled},
            "per_source": model.output.per_source,
        },
    }


def write_solve_json(model: Model, path: str | Path) -> None:
    with open(path, "w") as f:
        json.dump(model_to_solve_json(model), f, indent=1)
