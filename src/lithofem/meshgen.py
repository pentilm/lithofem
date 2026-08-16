"""Conformal tetrahedral mesh generation with Gmsh/OCC (docs/configuration.md, M3).

Pipeline: per-slab background boxes + per-frustum ruled lofts (mitre offsets
give a polytope, so a 2-section loft with matched vertex ordering is exact)
-> periodic wrap by intersecting translated copies with the cell ->
occ.fragment (conformal decomposition) -> centroid-based material/region
tagging (contract shared with config.model_to_solve_json) -> x/y periodic
surface pairing -> corner refinement (Distance+Threshold along frustum
corner trajectories) -> .msh 4.1.

Volume attributes are 1-based indices into the solve.json `regions` list.
Boundary attributes: 1 = PEC z-min (bottom of lower PML), 2 = PEC z-max,
3/4 = x-min/x-max sides, 5/6 = y-min/y-max sides.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import gmsh
import numpy as np

from .config import Model, model_to_solve_json
from .geometry import FrustumGeometry

BDR_PEC_ZMIN = 1
BDR_PEC_ZMAX = 2
BDR_XMIN, BDR_XMAX, BDR_YMIN, BDR_YMAX = 3, 4, 5, 6

# entity classification tolerance: OCC bounding boxes are padded by ~1e-7,
# so this must be comfortably larger (coordinates are nm-scale, slabs >> 1e-5)
_TOL = 1e-5


@dataclass
class MeshInfo:
    """Summary returned by generate(): region volumes as meshed (via OCC)."""

    path: str
    region_volumes: dict[int, float]  # attribute -> exact CAD volume
    n_regions: int


def mesh_info_from_file(model: Model, msh_path: str | Path) -> MeshInfo:
    """Rebuild the MeshInfo summary for a mesh that was not generated here.

    Used by the reuse paths (cached mesh, user-supplied mesh). Volumes are
    summed from the mesh itself rather than from CAD, so this doubles as a
    sanity check that the supplied mesh actually carries the region
    attributes the model expects.
    """
    from . import meshcheck

    stats = meshcheck.load_stats(str(msh_path))
    n_expected = len(model_to_solve_json(model)["regions"])
    missing = set(range(1, n_expected + 1)) - set(stats.region_volumes)
    if missing:
        raise ValueError(
            f"{msh_path} is missing region attributes {sorted(missing)}; the "
            f"model expects {n_expected} regions. Supply a mesh generated "
            "from this same configuration."
        )
    return MeshInfo(path=str(msh_path),
                    region_volumes=dict(stats.region_volumes),
                    n_regions=len(stats.region_volumes))


def mfem_mesh_path(msh_path: str | Path) -> Path:
    """Path of the derived msh 2.2 sibling consumed by the MFEM solver."""
    p = Path(msh_path)
    return p.with_suffix(".v22.msh")


def mfem_periodic_mesh_path(msh_path: str | Path) -> Path:
    """msh 2.2 sibling with a $Periodic section (vertex-identified in MFEM)."""
    return Path(msh_path).with_suffix(".per.msh")


def _periodic_node_pairs() -> list[tuple[int, int, int, int, list[tuple[int, int]]]]:
    """Collect (dim, slaveTag, masterTag, n, [(slave, master), ...]) blocks."""
    blocks = []
    for dim in (1, 2):
        for _, tag in gmsh.model.getEntities(dim):
            master, slave_nodes, master_nodes, _ = gmsh.model.mesh.getPeriodicNodes(
                dim, tag
            )
            if master == tag or len(slave_nodes) == 0:
                continue
            pairs = [(int(s), int(m)) for s, m in zip(slave_nodes, master_nodes)]
            blocks.append((dim, tag, int(master), len(pairs), pairs))
    return blocks


def _strip_periodic_section(path: Path) -> None:
    """Remove any $Periodic..$EndPeriodic section gmsh wrote into a 2.2 file."""
    lines = path.read_text().splitlines(keepends=True)
    out, skip = [], False
    for line in lines:
        if line.startswith("$Periodic"):
            skip = True
            continue
        if line.startswith("$EndPeriodic"):
            skip = False
            continue
        if not skip:
            out.append(line)
    path.write_text("".join(out))


def _append_periodic_section(path: Path, blocks: list) -> None:
    """Append a $Periodic section in the layout MFEM's 2.2 reader expects."""
    with open(path, "a") as f:
        f.write("$Periodic\n")
        f.write(f"{len(blocks)}\n")
        for dim, slave, master, n, pairs in blocks:
            f.write(f"{dim} {slave} {master}\n")
            f.write(f"{n}\n")
            for s, m in pairs:
                f.write(f"{s} {m}\n")
        f.write("$EndPeriodic\n")


def convert_to_v22(msh_path: str | Path) -> Path:
    """Convert a (possibly user-supplied) msh 4.1 file to the 2.2 sibling."""
    out = mfem_mesh_path(msh_path)
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(str(msh_path))
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.write(str(out))
    finally:
        gmsh.finalize()
    return out


def _loft_frustum(fr: FrustumGeometry) -> int:
    """Build one frustum solid as an exact polytope.

    Mitre offsets move each vertex along a straight line, so the solid is a
    polytope: bottom/top polygon faces plus one planar quad per edge. Built
    face-by-face (OCC addThruSections mismatches vertex correspondence and
    can produce twisted solids — see docs/gpu.md).
    """
    from .geometry import mitre_offset_vertices

    occ = gmsh.model.occ
    lo = mitre_offset_vertices(fr.base, fr.offset_at(fr.z_lo))
    hi = mitre_offset_vertices(fr.base, fr.offset_at(fr.z_hi))
    n = len(lo)
    p_lo = [occ.addPoint(x, y, fr.z_lo) for x, y in lo]
    p_hi = [occ.addPoint(x, y, fr.z_hi) for x, y in hi]
    l_lo = [occ.addLine(p_lo[i], p_lo[(i + 1) % n]) for i in range(n)]
    l_hi = [occ.addLine(p_hi[i], p_hi[(i + 1) % n]) for i in range(n)]
    l_vt = [occ.addLine(p_lo[i], p_hi[i]) for i in range(n)]
    faces = [
        occ.addPlaneSurface([occ.addCurveLoop(l_lo)]),
        occ.addPlaneSurface([occ.addCurveLoop(l_hi)]),
    ]
    for i in range(n):
        j = (i + 1) % n
        loop = occ.addCurveLoop([l_lo[i], l_vt[j], l_hi[i], l_vt[i]])
        faces.append(occ.addPlaneSurface([loop]))
    return occ.addVolume([occ.addSurfaceLoop(faces)])


def _solid_inside_cell(fr: FrustumGeometry, model: Model) -> bool:
    """True if the frustum (base + all offsets) stays within the cell."""
    from shapely.geometry import box as shp_box

    d_max = max(0.0, fr.offset_at(fr.z_lo), fr.offset_at(fr.z_hi))
    grown = fr.base.buffer(d_max + 1e-9, join_style="mitre", mitre_limit=1e9)
    return bool(shp_box(0, 0, model.domain.lx, model.domain.ly).covers(grown))


def _wrap_solid(vol: int, model: Model) -> list[int]:
    """Intersect periodic translations of the solid with the cell prism.

    Zero-volume touches (translations that only graze the cell boundary,
    e.g. for full-cell polygons) are dropped — they otherwise leave
    degenerate slivers that blow up the mesh (see design note).
    """
    dom = model.domain
    vol_tol = 1e-9 * dom.lx * dom.ly * (dom.z_max - dom.z_min)
    pieces: list[int] = []
    eps = 1.0
    for ix in (-1, 0, 1):
        for iy in (-1, 0, 1):
            box = gmsh.model.occ.addBox(
                0, 0, dom.z_min - eps, dom.lx, dom.ly, (dom.z_max - dom.z_min) + 2 * eps
            )
            (cp,) = [t for d, t in gmsh.model.occ.copy([(3, vol)]) if d == 3]
            gmsh.model.occ.translate([(3, cp)], ix * dom.lx, iy * dom.ly, 0)
            out, _ = gmsh.model.occ.intersect(
                [(3, cp)], [(3, box)], removeObject=True, removeTool=True
            )
            for d, t in out:
                if d != 3:
                    continue
                if gmsh.model.occ.getMass(d, t) < vol_tol:
                    gmsh.model.occ.remove([(3, t)], recursive=True)
                else:
                    pieces.append(t)
    gmsh.model.occ.remove([(3, vol)], recursive=True)
    return pieces


def _setup_periodicity(model: Model, z_lo: float, z_hi: float) -> None:
    """Pair x/y opposite boundary surfaces with translation maps."""
    dom = model.domain
    for axis, span in ((0, dom.lx), (1, dom.ly)):
        lo_surfs, hi_surfs = [], []
        for dim, tag in gmsh.model.getEntities(2):
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, tag)
            lo_v = (xmin, ymin)[axis]
            hi_v = (xmax, ymax)[axis]
            if abs(lo_v) < _TOL and abs(hi_v) < _TOL:
                lo_surfs.append(tag)
            elif abs(lo_v - span) < _TOL and abs(hi_v - span) < _TOL:
                hi_surfs.append(tag)
        if not lo_surfs:
            continue
        translation = [
            1, 0, 0, span if axis == 0 else 0,
            0, 1, 0, span if axis == 1 else 0,
            0, 0, 1, 0,
            0, 0, 0, 1,
        ]
        # match slave (hi) to master (lo) by shifted bounding box
        masters = []
        for h in hi_surfs:
            hb = np.array(gmsh.model.getBoundingBox(2, h))
            best, best_d = None, np.inf
            for lo in lo_surfs:
                lb = np.array(gmsh.model.getBoundingBox(2, lo))
                shift = np.zeros(6)
                shift[[0 + axis, 3 + axis]] = span
                d = float(np.max(np.abs(lb + shift - hb)))
                if d < best_d:
                    best, best_d = lo, d
            assert best is not None and best_d < 1e-5, f"no periodic partner for surf {h}"
            masters.append(best)
        gmsh.model.mesh.setPeriodic(2, hi_surfs, masters, translation)


def _boundary_attributes(model: Model, z_pml_lo: float, z_pml_hi: float) -> None:
    dom = model.domain
    groups: dict[int, list[int]] = {k: [] for k in range(1, 7)}
    for dim, tag in gmsh.model.getEntities(2):
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, tag)
        if abs(zmin - z_pml_lo) < _TOL and abs(zmax - z_pml_lo) < _TOL:
            groups[BDR_PEC_ZMIN].append(tag)
        elif abs(zmin - z_pml_hi) < _TOL and abs(zmax - z_pml_hi) < _TOL:
            groups[BDR_PEC_ZMAX].append(tag)
        elif abs(xmin) < _TOL and abs(xmax) < _TOL:
            groups[BDR_XMIN].append(tag)
        elif abs(xmin - dom.lx) < _TOL and abs(xmax - dom.lx) < _TOL:
            groups[BDR_XMAX].append(tag)
        elif abs(ymin) < _TOL and abs(ymax) < _TOL:
            groups[BDR_YMIN].append(tag)
        elif abs(ymin - dom.ly) < _TOL and abs(ymax - dom.ly) < _TOL:
            groups[BDR_YMAX].append(tag)
    names = {1: "pec_zmin", 2: "pec_zmax", 3: "xmin", 4: "xmax", 5: "ymin", 6: "ymax"}
    for attr, tags in groups.items():
        if tags:
            gmsh.model.addPhysicalGroup(2, tags, attr, name=names[attr])


def _corner_refinement(model: Model, base_size: float) -> None:
    """Distance+Threshold fields along frustum corner (vertex) trajectories."""
    if model.fem.corner_refine_radius is None or not model.frustums:
        return
    radius = model.fem.corner_refine_radius
    factor = model.fem.corner_refine_factor or 4.0
    from .geometry import mitre_offset_vertices

    pts: list[float] = []
    for f in model.frustums:
        lo_pts = mitre_offset_vertices(f.geom.base, f.geom.offset_at(f.geom.z_lo))
        hi_pts = mitre_offset_vertices(f.geom.base, f.geom.offset_at(f.geom.z_hi))
        for (xa, ya), (xb, yb) in zip(lo_pts, hi_pts):
            for t in np.linspace(0.0, 1.0, 9):
                x = xa + t * (xb - xa)
                y = ya + t * (yb - ya)
                z = f.geom.z_lo + t * (f.geom.z_hi - f.geom.z_lo)
                # wrap sample points into the cell for periodic cases
                if model.lateral_bc == "periodic":
                    x %= model.domain.lx
                    y %= model.domain.ly
                pts.extend((x, y, z))
    dist = gmsh.model.mesh.field.add("Distance")
    ptags = []
    for i in range(0, len(pts), 3):
        ptags.append(gmsh.model.occ.addPoint(pts[i], pts[i + 1], pts[i + 2]))
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.field.setNumbers(dist, "PointsList", ptags)
    thr = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(thr, "InField", dist)
    gmsh.model.mesh.field.setNumber(thr, "SizeMin", base_size / factor)
    gmsh.model.mesh.field.setNumber(thr, "SizeMax", base_size)
    gmsh.model.mesh.field.setNumber(thr, "DistMin", radius * 0.5)
    gmsh.model.mesh.field.setNumber(thr, "DistMax", radius)
    gmsh.model.mesh.field.setAsBackgroundMesh(thr)




def _check_periodic_safety(model: Model) -> None:
    """No element may touch both opposite periodic boundaries (see docs/gpu.md)."""
    tags, coords, _ = gmsh.model.mesh.getNodes()
    import numpy as _np

    xyz = _np.asarray(coords).reshape(-1, 3)
    idx = {int(t): i for i, t in enumerate(tags)}
    _, _, enodes = gmsh.model.mesh.getElements(3)
    for nn in enodes:
        conn = _np.asarray(nn).reshape(-1, 4)
        pts = _np.array([[idx[int(t)] for t in row] for row in conn])
        for axis, span in ((0, model.domain.lx), (1, model.domain.ly)):
            c = xyz[pts, axis]
            bad = (c.min(axis=1) < 1e-9) & (c.max(axis=1) > span - 1e-9)
            if bad.any():
                raise RuntimeError(
                    f"periodic-mesh safety: an element touches both "
                    f"boundaries along axis {axis}; the cell is too coarse "
                    f"for periodic identification — refine (smaller "
                    f"elems_per_wavelength target or larger cell)"
                )


def _set_mesh_algorithm(algorithm: str = "delaunay",
                        mesh_threads: int = 1) -> None:
    """Select the 3D meshing algorithm and thread count.

    The default is gmsh's classic Delaunay (Algorithm3D = 1) - deliberately.
    HXT (= 10) meshes the same geometry about twice as fast and with ~10%
    fewer elements, and passes every geometric equivalence check, but at
    equal target size those fewer elements cost accuracy exactly where TM
    polarization is hardest (field jumps at material interfaces): the M8-1
    TM-vs-RCWA acceptance case degrades from 5.9e-4 / 1.6e-4 to 1.66e-3 /
    1.11e-3, through its 1e-3 gate (measured against the independent RCWA
    reference in the test suite). Meshing is ~1 s of a
    ~55 s solve, so the default buys reproducible accuracy with the speed of
    the mesher being a non-factor; `fem.mesh_algorithm: hxt` remains
    available for large geometries where meshing time actually matters.

    Threads default to 1 because gmsh's threaded meshing (either algorithm)
    is not run-to-run reproducible, and a mesh that moves drags the
    discretization error with it. `fem.mesh_threads` opts in.

    Environment overrides (LITHOFEM_MESH_ALGO3D numeric, LITHOFEM_MESH_THREADS)
    take precedence; the regression tests use them for A/B comparisons.
    """
    default_algo = "10" if algorithm == "hxt" else "1"
    algo = int(os.environ.get("LITHOFEM_MESH_ALGO3D", default_algo))
    threads = int(os.environ.get("LITHOFEM_MESH_THREADS", str(mesh_threads)))
    gmsh.option.setNumber("Mesh.Algorithm3D", algo)
    gmsh.option.setNumber("General.NumThreads", threads)
    if algo == 10:
        gmsh.option.setNumber("Mesh.MaxNumThreads3D", threads)


def generate(model: Model, out_path: str | Path, verbose: bool = False) -> MeshInfo:
    """Generate mesh.msh (4.1) for the expanded model. Returns CAD volumes."""
    doc = model_to_solve_json(model)
    regions = doc["regions"]
    dom = model.domain
    t_pml = model.pml.thickness_wavelengths * model.wavelength
    z_pml_lo = dom.z_min - t_pml
    z_pml_hi = dom.z_max + t_pml

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
        gmsh.model.add("lithofem")

        n_slab = len(model.slabs) - 1
        solids: list[int] = []
        owner: list[tuple[str, int]] = []  # ("slab", i) | ("pml", 0/1) | ("frustum", j)
        for i in range(n_slab):
            solids.append(gmsh.model.occ.addBox(
                0, 0, model.slabs[i], dom.lx, dom.ly, model.slabs[i + 1] - model.slabs[i]
            ))
            owner.append(("slab", i))
        solids.append(gmsh.model.occ.addBox(0, 0, z_pml_lo, dom.lx, dom.ly, t_pml))
        owner.append(("pml", 0))
        solids.append(gmsh.model.occ.addBox(0, 0, dom.z_max, dom.lx, dom.ly, t_pml))
        owner.append(("pml", 1))
        for j, f in enumerate(model.frustums):
            vol = _loft_frustum(f.geom)
            need_wrap = (
                model.lateral_bc == "periodic"
                and not _solid_inside_cell(f.geom, model)
            )
            pieces = _wrap_solid(vol, model) if need_wrap else [vol]
            for p in pieces:
                solids.append(p)
                owner.append(("frustum", j))

        dimtags = [(3, s) for s in solids]
        _, out_map = gmsh.model.occ.fragment(dimtags, [])
        gmsh.model.occ.synchronize()

        # children per input; frustum ownership wins over the slab boxes
        frustum_of: dict[int, int] = {}
        slab_of: dict[int, int] = {}
        pml_of: dict[int, int] = {}
        for (kind, idx), children in zip(owner, out_map):
            for d, child in children:
                if d != 3:
                    continue
                if kind == "frustum":
                    frustum_of[child] = idx
                elif kind == "slab":
                    slab_of[child] = idx
                else:
                    pml_of[child] = idx

        def region_attr(tag: int) -> int:
            if tag in pml_of:
                kind = "pml_bottom" if pml_of[tag] == 0 else "pml_top"
                for k, reg in enumerate(regions):
                    if reg["kind"] == kind:
                        return k + 1
            if tag in frustum_of:
                j = frustum_of[tag]
                _, _, z = gmsh.model.occ.getCenterOfMass(3, tag)
                i = max(0, min(int(np.searchsorted(model.slabs, z) - 1), n_slab - 1))
                for k, reg in enumerate(regions):
                    if reg["kind"] == "frustum" and reg["frustum"] == j and reg["slab"] == i:
                        return k + 1
            if tag in slab_of:
                for k, reg in enumerate(regions):
                    if reg["kind"] == "background" and reg["slab"] == slab_of[tag]:
                        return k + 1
            raise RuntimeError(f"cannot classify volume {tag}")

        vols_by_region: dict[int, list[int]] = {}
        region_volumes: dict[int, float] = {}
        for dim, tag in gmsh.model.getEntities(3):
            attr = region_attr(tag)
            vols_by_region.setdefault(attr, []).append(tag)
            region_volumes[attr] = region_volumes.get(attr, 0.0) + gmsh.model.occ.getMass(
                dim, tag
            )
        for attr, tags in sorted(vols_by_region.items()):
            gmsh.model.addPhysicalGroup(3, tags, attr, name=f"region_{attr}")

        _boundary_attributes(model, z_pml_lo, z_pml_hi)

        if model.lateral_bc == "periodic":
            _setup_periodicity(model, z_pml_lo, z_pml_hi)

        # mesh sizes: lambda / (n * elems_per_wavelength) using the largest
        # real refractive index present (conservative global size)
        n_max = 1.0
        for reg in regions:
            e = complex(reg["epsilon"][0], reg["epsilon"][1])
            n_max = max(n_max, abs(np.sqrt(e).real), abs(np.sqrt(e)))
        base_size = model.wavelength / (model.fem.elems_per_wavelength * n_max)
        if model.lateral_bc == "periodic":
            # periodic vertex identification needs enough element layers
            # across each periodic direction (else opposite-boundary faces
            # collide after identification; see docs/gpu.md). Four layers
            # proved borderline with fragmented geometries -> use five.
            base_size = min(base_size, dom.lx / 5.0, dom.ly / 5.0)
        gmsh.option.setNumber("Mesh.MeshSizeMax", base_size)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)

        _corner_refinement(model, base_size)

        _set_mesh_algorithm(model.fem.mesh_algorithm,
                            model.fem.mesh_threads)
        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.renumberNodes()  # contiguous tags == file numbering

        if model.lateral_bc == "periodic":
            _check_periodic_safety(model)

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # contract file: msh 4.1; derived sibling: msh 2.2 for MFEM, whose
        # gmsh reader only parses the 2.2 layout (docs/gpu.md)
        gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
        gmsh.write(str(out_path))
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        v22_path = mfem_mesh_path(out_path)
        gmsh.write(str(v22_path))
        # the plain solver mesh must NOT identify periodic vertices
        _strip_periodic_section(v22_path)

        if model.lateral_bc == "periodic":
            # third sibling: 2.2 + $Periodic (vertex-identified in MFEM), with
            # side-surface physical groups dropped so no boundary elements sit
            # on the periodically identified faces (docs/gpu.md)
            blocks = _periodic_node_pairs()
            gmsh.model.removePhysicalGroups(
                [(2, a) for a in (BDR_XMIN, BDR_XMAX, BDR_YMIN, BDR_YMAX)]
            )
            per_path = mfem_periodic_mesh_path(out_path)
            gmsh.write(str(per_path))
            _strip_periodic_section(per_path)
            _append_periodic_section(per_path, blocks)
        return MeshInfo(
            path=str(out_path),
            region_volumes=region_volumes,
            n_regions=len(regions),
        )
    finally:
        gmsh.finalize()


# --------------------------------------------------------------------------
# analytic volumes (for the M3 conservation test)
# --------------------------------------------------------------------------


def analytic_region_volumes(model: Model) -> dict[int, float]:
    """Exact volume of every region (Simpson per slab: A(z) is quadratic)."""
    doc = model_to_solve_json(model)
    regions = doc["regions"]
    dom = model.domain
    cell_area = dom.lx * dom.ly
    t_pml = model.pml.thickness_wavelengths * model.wavelength
    vols: dict[int, float] = {}

    def frustum_slab_volume(f: FrustumGeometry, z_lo: float, z_hi: float) -> float:
        zs = (z_lo, 0.5 * (z_lo + z_hi), z_hi)
        areas = []
        for z in zs:
            cs = f.cross_section(z)
            assert cs is not None
            areas.append(cs.area)
        return (z_hi - z_lo) / 6.0 * (areas[0] + 4 * areas[1] + areas[2])

    for k, reg in enumerate(regions):
        attr = k + 1
        if reg["kind"] in ("pml_bottom", "pml_top"):
            vols[attr] = cell_area * t_pml
            continue
        i = reg["slab"]
        z_lo, z_hi = model.slabs[i], model.slabs[i + 1]
        if reg["kind"] == "frustum":
            f = model.frustums[reg["frustum"]].geom
            a, b = max(z_lo, f.z_lo), min(z_hi, f.z_hi)
            vols[attr] = frustum_slab_volume(f, a, b) if b > a else 0.0
        else:  # background = slab minus all frustums present in the slab
            v = cell_area * (z_hi - z_lo)
            zc = 0.5 * (z_lo + z_hi)
            for fr in model.frustums:
                if fr.geom.z_lo <= zc <= fr.geom.z_hi:
                    a, b = max(z_lo, fr.geom.z_lo), min(z_hi, fr.geom.z_hi)
                    if b > a:
                        v -= frustum_slab_volume(fr.geom, a, b)
            vols[attr] = v
    return vols
