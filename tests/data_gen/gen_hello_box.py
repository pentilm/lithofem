"""Generate a simple tagged tet mesh of a box (msh 4.1) for the M0 hello-world.

Deterministic: fixed characteristic length, no randomness.
Box: 100 x 100 x 50 nm, one physical volume (tag 1), analytic volume 5e5 nm^3.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gmsh

BOX = (100.0, 100.0, 50.0)
ANALYTIC_VOLUME = BOX[0] * BOX[1] * BOX[2]


def generate(out: Path, lc: float = 20.0) -> None:
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("hello_box")
        tag = gmsh.model.occ.addBox(0, 0, 0, *BOX)
        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(3, [tag], 1, name="box")
        boundary = [abs(s) for _, s in gmsh.model.getBoundary([(3, tag)], oriented=False)]
        gmsh.model.addPhysicalGroup(2, boundary, 1, name="walls")
        gmsh.option.setNumber("Mesh.MeshSizeMax", lc)
        gmsh.model.mesh.generate(3)
        out.parent.mkdir(parents=True, exist_ok=True)
        gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
        gmsh.write(str(out))
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.write(str(out.with_suffix(".v22.msh")))
    finally:
        gmsh.finalize()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    args = ap.parse_args()
    generate(args.out)


if __name__ == "__main__":
    main()
