# Third-party components and licensing

LithoFEM's own source is licensed under Apache-2.0 (see `LICENSE`). It builds on
several third-party projects whose licenses differ; this matters most if you
plan to **redistribute binaries** or embed LithoFEM in a closed-source product.

| Component | Role | License |
|---|---|---|
| [MFEM](https://mfem.org) ≥ 4.9 | finite element library (assembly, spaces, meshes) | BSD-3-Clause |
| [Gmsh](https://gmsh.info) (Python API) | geometry construction and tetrahedral meshing | **GPL-2.0-or-later** |
| [SuiteSparse](https://people.engr.tamu.edu/davis/suitesparse.html) — UMFPACK, CHOLMOD | CPU complex sparse direct solve, fill-in estimation | **GPL-2.0-or-later** (AMD/COLAMD components are BSD) |
| [NVIDIA cuDSS](https://developer.nvidia.com/cudss) | GPU sparse direct solve | proprietary (NVIDIA Software License Agreement) |
| [nlohmann/json](https://github.com/nlohmann/json) (vendored, `solver/thirdparty/json.hpp`) | JSON parsing in the C++ solver | MIT |
| numpy, scipy, shapely, h5py, PyYAML | Python support libraries | BSD-3-Clause / MIT |
| pyvista (optional) | ParaView output inspection in tests | MIT |

## Practical notes

- **Distributing LithoFEM source** under Apache-2.0 is straightforward: the GPL
  components are separate upstream projects that a user installs themselves, and
  no GPL code is included in this repository.
- **Distributing compiled binaries** that link SuiteSparse, or bundling Gmsh,
  brings the GPL obligations into play. If you need a closed-source binary
  distribution you would have to replace Gmsh (meshing) and UMFPACK/CHOLMOD
  (CPU solve and memory estimation) with differently licensed equivalents. Note
  that the GPU path (cuDSS) does not depend on SuiteSparse, but the automatic
  CPU fallback does.
- **cuDSS** is redistributed by NVIDIA under its own terms; LithoFEM only
  detects and links an installation the user provides (via the
  `nvidia-cudss-cu13` wheel or a system install).

None of this restricts using LithoFEM for research or in-house work, including
commercially.
