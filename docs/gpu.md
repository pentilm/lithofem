# GPU acceleration

LithoFEM can run both dominant stages — matrix assembly and the sparse direct
solve — entirely on the GPU, with the assembled matrix handed to the solver
without ever travelling through host memory.

```yaml
solver: {type: direct, device: gpu, gpu_ids: [0]}
fem:    {assembly: gpu}
```

Both switches default to `cpu` and are independent; the CPU path is unchanged
and always available as a fallback.

## GPU direct solve (cuDSS)

With `solver.device: gpu` and `type: direct`, the complex system is factorized
by NVIDIA cuDSS. The factors stay cached on the device, so multi-source runs
(`output.per_source: true`) pay for one factorization and reuse it for every
right-hand side.

Notes:

- The matrix type is always **general**, never complex-symmetric. At oblique
  incidence the Bloch envelope substitution makes the system genuinely
  non-symmetric; a symmetric setting produces a silently wrong solution because
  the library reads only one triangle without checking.
- Before factorizing, the analysis phase's memory estimate is compared against
  the available VRAM (or `solver.gpu_mem_gb` / `--gpu-mem-limit-gb`). If the
  factors would not fit, the run logs the numbers and falls back to the CPU
  solver rather than crashing. The estimate has tracked the measured peak to
  within about 5% across problem sizes.
- Any failure — no device, allocation failure, numerical breakdown — logs an
  explicit message and falls back to UMFPACK on the CPU.
- When the direct GPU solver is used, MFEM's own device is deliberately left on
  CPU: the host-side right-hand side and sampling paths are the ones covered by
  the validation suite, and cuDSS does not need MFEM's device runtime.

## GPU matrix assembly

With `fem.assembly: gpu` (which requires the cuDSS path above), the system
matrix is built by CUDA kernels written for this purpose:

1. **Host extraction, once per mesh** — element DOF maps and signs, face
   orientations for the H(curl) orientation transform, affine element geometry
   (constant Jacobian, inverse, determinant), the region coefficient table,
   reference basis and curl tables at the quadrature points, and the symbolic
   CSR structure.
2. **Element matrices on device** — one block per element; the five integrators
   of the sesquilinear form (curl-curl, mass, two mixed curl couplings, and the
   `k∥` mass term) are evaluated in complex double precision, with the PML
   stretch evaluated at each quadrature point. Physical basis values are staged
   in shared memory per quadrature point.
3. **Orientation transform** — the H(curl) face-DOF transform is applied on
   device as small 2×2 blocks, in the same order as the reference implementation.
4. **Scatter and elimination** — values are accumulated into the CSR with a
   binary search over the sorted row plus `atomicAdd`, then essential (PEC) rows
   and columns are eliminated by a device kernel.
5. **Zero-copy handoff** — cuDSS is given the device CSR pointers directly.

The self-residual is computed by a device SpMV, so the matrix never needs to
exist on the host at all.

### Correctness

GPU assembly is required to reproduce the CPU discretization **element by
element**, not merely to converge to the same answer. The test suite
(`tests/test_asm_v25.py`) compares, at a relative tolerance of 1e-13:

| Layer | Compared against | Coverage |
|---|---|---|
| Reference basis/curl tables | direct library calls | p = 1…4, all quadrature points |
| Affine geometry | element transformations | sampled elements + total mesh volume |
| Element matrices | per-integrator reference matrices | 5 integrators × p = 1…4 × region types × 2 incidence angles |
| Orientation transform | reference transform | every element of the mesh, p = 2…4 |
| Global CSR | CPU-assembled matrix | 3 geometries × p = 1…3 × 2 angles; bitwise-identical sparsity |
| Eliminated system + RHS | CPU-formed linear system | same cases |
| End-to-end | CPU-assembly run | fields < 1e-10, order efficiencies < 1e-8 |

Measured worst cases are around 2–4e-15, i.e. more than two orders of magnitude
inside the criterion.

### Portability

The kernels use only standard CUDA C++: fp64 arithmetic, shared memory,
`__constant__` tables and `atomicAdd(double)` (sm_60 and newer). Complex numbers
are a plain two-double struct. There are no `__CUDA_ARCH__` branches, no tensor
cores, no architecture-specific instructions — the target is set purely by the
build flag. See [building.md](building.md#6-gpu-architectures-and-portability)
for multi-architecture (fatbin) builds.

## Multi-GPU parameter sweeps

Independent solve groups are distributed across GPUs at the task level, one
process per task with a pinned device:

```yaml
sweep: {gpu_ids: [0, 1], max_parallel: 2}
```

There is no inter-process communication; each task solves a complete problem.
Results are bitwise comparable to running the groups serially. A failing task is
isolated: the others complete, and the summary points at the failing group's log.

Measured on two NVIDIA H200s: three tasks over two cards gives a 1.44× wall-clock
speedup (the scheduling ceiling for that shape is 1.5×).

A *single* problem always runs on a single GPU. There is no MPI or domain
decomposition, so the largest solvable problem is bounded by one card's memory.

## Measured performance

EUV line grating, cubic Nédélec elements, 6° incidence, single NVIDIA H200.
Times in seconds, from the per-stage timings written into the run metadata.

| DOF (complex) | Assembly (CPU) | Assembly (GPU) | Speedup | Solve (cuDSS) | End-to-end (GPU) |
|---|---|---|---|---|---|
| 52,167 | 1.0 | 0.1 | 15× | 0.5 | 1.4 |
| 460,670 | 12.0 | 0.6 | 20× | 6.0 | 11.9 |
| 1,344,564 | 103 | 3.9 | 26× | 8.4 | 55 |

For scale, the same 1.34M-DOF problem on the pure-CPU path (serial assembly plus
UMFPACK) takes 438 s end-to-end.

At this point the GPU stages are no longer the bottleneck: in the 55 s run,
post-processing (field sampling and per-region integrals) and right-hand-side
assembly — both single-threaded host code — account for over half the time.

Reproduce with:

```bash
python tools/benchmark.py  [outdir]        # CPU vs GPU assembly, three sizes
python tools/profile_segments.py [outdir]  # per-stage timing breakdown
```

## Memory bounds

A sparse direct factorization is memory-hungry: the 1.34M-DOF case above needs
about 35 GB of device memory for its factors. On a 140 GB card that puts the
practical ceiling at a few million DOF. Beyond that you would need an iterative
solver or a distributed direct solver, neither of which LithoFEM currently has.
