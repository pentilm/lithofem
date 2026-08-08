"""Polygon and frustum geometry processing (docs/configuration.md).

All lengths in nm. Frustum cross-section convention (see docs/configuration.md): the base polygon lives at z0; at height z the cross
section is the base offset by  d(z) = -|z - z0| / tan(alpha).  alpha is the
angle between the side wall and the xy plane: alpha < 90 deg tapers away
from the base (typical etched profile), alpha = 90 is vertical, alpha > 90
expands. This holds for both signs of h (the solid always tapers/expands
away from its base plane).

Mitre offsetting keeps corners sharp (planar side walls -> polytope), so
each base vertex moves along its angle bisector at constant rate; validity
over the full z-range is checked and, on collapse, the maximum legal |h| is
computed by bisection.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from shapely.geometry import Polygon
from shapely.validation import explain_validity

# Effectively-unlimited mitre: corners are never bevelled (true planar walls).
_MITRE_LIMIT = 1e9
# Relative area tolerance used in wrap conservation and overlap checks.
AREA_RTOL = 1e-10


class GeometryError(ValueError):
    """Raised for invalid polygon/frustum geometry; message includes fixes."""


def polygon_from_vertices(vertices: list[tuple[float, float]] | np.ndarray) -> Polygon:
    """Validate a simple polygon and normalize orientation to CCW.

    Raises GeometryError (with explanation) for < 3 vertices, repeated
    consecutive vertices, zero area, or self-intersection.
    """
    pts = np.asarray(vertices, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 3:
        raise GeometryError("polygon needs >= 3 (x, y) vertices")
    if np.linalg.norm(pts[0] - pts[-1]) < 1e-12:
        pts = pts[:-1]  # tolerate explicitly closed input
        if pts.shape[0] < 3:
            raise GeometryError("polygon needs >= 3 distinct vertices")
    dup = np.linalg.norm(np.diff(np.vstack([pts, pts[:1]]), axis=0), axis=1) < 1e-12
    if dup.any():
        idx = int(np.nonzero(dup)[0][0])
        raise GeometryError(f"repeated consecutive vertex at index {idx}; remove the duplicate")
    poly = Polygon(pts)
    if not poly.is_valid or not poly.is_simple:
        raise GeometryError(
            f"polygon is not simple: {explain_validity(poly)}; "
            "fix the self-intersection or reorder the vertices"
        )
    if poly.area <= 0:
        raise GeometryError("polygon has zero area")
    # normalize to CCW exterior
    if not poly.exterior.is_ccw:
        poly = Polygon(list(poly.exterior.coords)[::-1])
    return poly


def mitre_offset(poly: Polygon, dist: float) -> Polygon | None:
    """Uniform mitre offset by signed dist (positive = outward).

    Returns the offset polygon, or None if the offset collapses or changes
    topology (splits into pieces, develops holes, loses/gains corners in a
    way that breaks the single-simple-polygon requirement).
    """
    if dist == 0.0:
        return poly
    out = poly.buffer(dist, join_style="mitre", mitre_limit=_MITRE_LIMIT)
    if out.is_empty or out.geom_type != "Polygon" or len(out.interiors) > 0:
        return None
    if not out.is_valid or out.area <= 0:
        return None
    # topology preservation: mitre offset of a fixed-topology polygon keeps
    # one moving vertex per corner (collinear-vertex merging aside)
    n_in = len(poly.exterior.coords) - 1
    n_out = len(out.exterior.coords) - 1
    if n_out != n_in:
        return None
    return out


@dataclass(frozen=True)
class FrustumGeometry:
    """Validated frustum: CCW base polygon at z0, extent h (signed), alpha deg."""

    base: Polygon
    z0: float
    h: float
    alpha: float
    index: int = -1  # position in the user's frustums list, for error messages

    @property
    def z_lo(self) -> float:
        return min(self.z0, self.z0 + self.h)

    @property
    def z_hi(self) -> float:
        return max(self.z0, self.z0 + self.h)

    def offset_at(self, z: float) -> float:
        """Signed offset distance of the cross-section at height z."""
        return -abs(z - self.z0) / np.tan(np.deg2rad(self.alpha))

    def cross_section(self, z: float) -> Polygon | None:
        return mitre_offset(self.base, self.offset_at(z))


def validate_frustum_extrusion(fr: FrustumGeometry, n_check: int = 33) -> None:
    """Check the mitre offset stays a single simple polygon over the z-range.

    On failure raises GeometryError reporting the maximum legal |h| found by
    bisection (relative precision 1e-3, i.e. well within the 1% acceptance).
    """
    if fr.alpha == 90.0 or fr.h == 0.0:
        return
    zs = np.linspace(fr.z_lo, fr.z_hi, n_check)
    if all(fr.cross_section(z) is not None for z in zs):
        return
    # bisection on |h|: valid(t) := extrusion of height t*|h| from base is OK
    def ok(habs: float) -> bool:
        rate = 1.0 / np.tan(np.deg2rad(fr.alpha))
        return all(
            mitre_offset(fr.base, -t * rate) is not None
            for t in np.linspace(0.0, habs, n_check)
        )

    lo, hi = 0.0, abs(fr.h)
    if not ok(0.0):
        raise GeometryError(
            f"frustum[{fr.index}]: base polygon invalid for offsetting"
        )
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if ok(mid):
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-4 * abs(fr.h):
            break
    raise GeometryError(
        f"frustum[{fr.index}]: mitre offset collapses within |h|={abs(fr.h):g} "
        f"(alpha={fr.alpha:g} deg); maximum allowed |h| is {lo:.6g} nm — "
        "reduce |h|, increase alpha toward 90, or simplify the polygon"
    )


def max_legal_h(base: Polygon, alpha: float, h_cap: float, n_check: int = 129) -> float:
    """Reference bisection for the maximum legal |h| (used by tests as truth)."""
    rate = 1.0 / np.tan(np.deg2rad(alpha))

    def ok(habs: float) -> bool:
        return all(
            mitre_offset(base, -t * rate) is not None
            for t in np.linspace(0.0, habs, n_check)
        )

    lo, hi = 0.0, h_cap
    if ok(h_cap):
        return h_cap
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if ok(mid):
            lo = mid
        else:
            hi = mid
    return lo


def mitre_offset_vertices(poly: Polygon, dist: float) -> np.ndarray:
    """Exact mitre-offset vertex positions with 1:1 correspondence.

    v_i' = v_i + dist * (n_{i-1} + n_i) / (1 + n_{i-1}.n_i), with n_i the
    outward unit normal of edge (v_i, v_{i+1}) of the CCW polygon. Used by
    the mesh generator to build frustum polytopes with matched rings (shapely
    buffer may renumber vertices; this never does). Validity of the offset
    must be established separately via mitre_offset().
    """
    pts = np.asarray(poly.exterior.coords)[:-1]
    e = np.roll(pts, -1, axis=0) - pts
    en = e / np.linalg.norm(e, axis=1, keepdims=True)
    nor = np.stack([en[:, 1], -en[:, 0]], axis=1)  # outward for CCW
    out = np.empty_like(pts)
    for i in range(len(pts)):
        n_prev, n_cur = nor[i - 1], nor[i]
        denom = 1.0 + float(np.dot(n_prev, n_cur))
        out[i] = pts[i] + dist * (n_prev + n_cur) / denom
    return out


def wrap_polygon(poly: Polygon, lx: float, ly: float) -> list[Polygon]:
    """Wrap a polygon into the periodic cell [0,Lx]x[0,Ly].

    The polygon may overhang any cell boundary (but must fit in one period).
    Returns the pieces inside the cell after periodic translation; total area
    is conserved to AREA_RTOL (checked by the caller/tests).
    """
    cell = Polygon([(0, 0), (lx, 0), (lx, ly), (0, ly)])
    pieces: list[Polygon] = []
    for ix in (-1, 0, 1):
        for iy in (-1, 0, 1):
            shifted = _translate(poly, ix * lx, iy * ly)
            inter = shifted.intersection(cell)
            if inter.is_empty:
                continue
            geoms = getattr(inter, "geoms", [inter])
            pieces.extend(g for g in geoms if g.geom_type == "Polygon" and g.area > 0)
    return pieces


def _translate(poly: Polygon, dx: float, dy: float) -> Polygon:
    pts = np.asarray(poly.exterior.coords)[:-1] + np.array([dx, dy])
    return Polygon(pts)


def z_breakpoints(
    z_min: float,
    z_max: float,
    layer_bounds: list[float],
    frustum_bounds: list[float],
    obs_planes: list[float],
    tol: float = 1e-9,
) -> list[float]:
    """Sorted unique z breakpoints inside [z_min, z_max] -> slab boundaries."""
    zs = [z_min, z_max]
    for z in [*layer_bounds, *frustum_bounds, *obs_planes]:
        if z_min - tol < z < z_max + tol:
            zs.append(float(np.clip(z, z_min, z_max)))
    zs.sort()
    out: list[float] = []
    for z in zs:
        if not out or z - out[-1] > tol:
            out.append(z)
    if abs(out[-1] - z_max) > tol:
        out.append(z_max)
    else:
        out[-1] = z_max
    out[0] = z_min
    return out


def frustum_overlap_errors(
    frustums: list[FrustumGeometry],
    lx: float | None,
    ly: float | None,
    breakpoints: list[float],
) -> list[str]:
    """Pairwise frustum-overlap detection; returns error strings.

    Cross sections are compared at every slab boundary and slab midpoint of
    the common z-interval (wrapped into the cell when periodic). Sampling at
    slab resolution is exact for the piecewise-planar geometry used here in
    all practical cases (see docs/gpu.md).
    """
    errors: list[str] = []
    for i in range(len(frustums)):
        for j in range(i + 1, len(frustums)):
            fi, fj = frustums[i], frustums[j]
            lo = max(fi.z_lo, fj.z_lo)
            hi = min(fi.z_hi, fj.z_hi)
            if hi - lo <= 1e-12:
                continue
            zs = [z for z in breakpoints if lo - 1e-12 <= z <= hi + 1e-12]
            if not zs:
                zs = [lo, hi]
            samples = sorted(set(zs) | {0.5 * (a + b) for a, b in zip(zs, zs[1:])})
            for z in samples:
                zc = float(np.clip(z, lo, hi))
                pi, pj = fi.cross_section(zc), fj.cross_section(zc)
                if pi is None or pj is None:
                    continue  # collapse reported separately
                if lx is not None and ly is not None:
                    pieces_i = wrap_polygon(pi, lx, ly)
                    pieces_j = wrap_polygon(pj, lx, ly)
                else:
                    pieces_i, pieces_j = [pi], [pj]
                area = sum(a.intersection(b).area for a in pieces_i for b in pieces_j)
                if area > AREA_RTOL * max(pi.area, pj.area):
                    errors.append(
                        f"frustums[{i}] and frustums[{j}] overlap at z={zc:g} "
                        f"(intersection area {area:g} nm^2); separate the "
                        "polygons or adjust z0/h so their z-ranges do not meet"
                    )
                    break
    return errors
