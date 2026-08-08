"""Mesh verification utilities for M3 acceptance (pure Python, reads .msh).

Checks: per-region tet volume sums, conformity (no hanging nodes), positive
Jacobians, periodic boundary node pairing, element size statistics.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import gmsh
import numpy as np


@dataclass
class MeshStats:
    region_volumes: dict[int, float]
    min_jacobian: float
    n_hanging_faces: int
    n_tets: int
    nodes: np.ndarray  # (n, 3) coordinates
    tets: np.ndarray  # (m, 4) 0-based node indices
    tet_attrs: np.ndarray  # (m,) region attribute per tet


def load_stats(msh_path: str) -> MeshStats:
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(msh_path)
        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        order = np.argsort(node_tags)
        node_tags = np.asarray(node_tags)[order]
        xyz = np.asarray(coords).reshape(-1, 3)[order]
        tag_to_idx = {int(t): i for i, t in enumerate(node_tags)}

        tets_list: list[np.ndarray] = []
        attrs_list: list[np.ndarray] = []
        for dim, phys in gmsh.model.getPhysicalGroups(3):
            for ent in gmsh.model.getEntitiesForPhysicalGroup(dim, phys):
                etypes, etags, enodes = gmsh.model.mesh.getElements(3, ent)
                for et, tags_e, nn in zip(etypes, etags, enodes):
                    if et != 4:  # linear tet
                        continue
                    conn = np.asarray(nn).reshape(-1, 4)
                    tets_list.append(conn)
                    attrs_list.append(np.full(len(conn), phys))
        tets_raw = np.vstack(tets_list)
        attrs = np.concatenate(attrs_list)
        tets = np.vectorize(tag_to_idx.__getitem__)(tets_raw.astype(int))
    finally:
        gmsh.finalize()

    p = xyz[tets]  # (m, 4, 3)
    v6 = np.einsum(
        "ij,ij->i",
        np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]),
        p[:, 3] - p[:, 0],
    )
    vols = v6 / 6.0

    region_volumes: dict[int, float] = {}
    for a in np.unique(attrs):
        region_volumes[int(a)] = float(np.abs(vols[attrs == a]).sum())

    # conformity: each interior tet face shared by exactly 2 tets, boundary by 1
    face_count: dict[tuple[int, ...], int] = defaultdict(int)
    faces_idx = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    for tet in tets:
        for fi in faces_idx:
            key = tuple(sorted(int(tet[k]) for k in fi))  # noqa: RUF015
            face_count[key] += 1
    hanging = sum(1 for c in face_count.values() if c > 2)

    return MeshStats(
        region_volumes=region_volumes,
        min_jacobian=float(np.min(v6)),
        n_hanging_faces=hanging,
        n_tets=len(tets),
        nodes=xyz,
        tets=tets,
        tet_attrs=attrs,
    )


def signed_jacobians(stats: MeshStats) -> np.ndarray:
    p = stats.nodes[stats.tets]
    return np.einsum(
        "ij,ij->i",
        np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]),
        p[:, 3] - p[:, 0],
    )


def periodic_pairing_error(
    stats: MeshStats, lx: float, ly: float, tol_snap: float = 1e-6
) -> float:
    """Max distance between each boundary node and its periodic partner.

    Boundary nodes at x=0 must map onto nodes at x=lx (same y, z), same for y.
    Returns the worst mismatch over both directions (inf if a node is
    unmatched).
    """
    xyz = stats.nodes
    worst = 0.0
    for axis, span in ((0, lx), (1, ly)):
        lo_mask = np.abs(xyz[:, axis]) < tol_snap
        hi_mask = np.abs(xyz[:, axis] - span) < tol_snap
        lo = xyz[lo_mask].copy()
        hi = xyz[hi_mask].copy()
        if len(lo) != len(hi):
            return float("inf")
        lo_shift = lo.copy()
        lo_shift[:, axis] += span
        from scipy.spatial import cKDTree

        d_ab, _ = cKDTree(lo_shift).query(hi)
        d_ba, _ = cKDTree(hi).query(lo_shift)
        worst = max(worst, float(np.max(d_ab)), float(np.max(d_ba)))
    return worst
