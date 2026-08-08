# Validation

LithoFEM is validated layer by layer: every component is pinned to a reference
that does not share its implementation, and each check is a regression test in
`tests/`. The chain is deliberately ordered so that each stage only depends on
stages already verified.

```
analytic TMM ──► assembly core (MMS) ──► PML ──► end-to-end vs TMM
                                                      │
                          local sources vs Green's fn ─┤
                                                      ▼
                                    patterned gratings vs independent RCWA
                                                      │
                                          GPU paths ──┘ (element-wise vs CPU)
```

## Layer 1 — transfer-matrix reference

An independent TMM implementation (`src/lithofem/tmm.py`) is verified against
closed-form Fresnel results for single interfaces, absorbing media, and
symmetry/reciprocity relations, to machine precision. It is then trusted as the
analytic reference for the layers above, and used at run time to supply the
incident field for the scattered-field formulation.

Tests: `tests/test_tmm.py`.

## Layer 2 — assembly core, method of manufactured solutions

A known analytic field is substituted into the governing equation to produce a
source term; the discrete solve must recover the field at the theoretical
convergence rate. This checks the complex curl-curl assembly, the sesquilinear
form, boundary conditions and the direct solver together, without any physics
modelling assumptions.

Verified for Nédélec orders p = 1…6, with the observed convergence order matching
theory in each case.

Tests: `tests/test_mms.py`, binary `solver/mms_test.cpp`.

## Layer 3 — PML

Reflection from the absorbing layer is measured directly by comparing a solve
against the analytic outgoing solution, over a sweep of thickness and profile
order. Reflection stays below 1e-6 in the useful parameter range.

Tests: `tests/test_pml.py`; the parameter sweep tool is `tools/pml_scan.py`.

## Layer 4 — end-to-end against the analytic stack

A layered stack with one layer expressed as a full-cell frustum is solved with
the complete pipeline and compared against the TMM solution of the equivalent
stack. This exercises meshing, periodicity, PML, the scattered-field source, the
solve and the field extraction as a whole.

Two complementary results:

- **Zero-scattering test** — an unpatterned stack must produce exactly zero
  scattered field. Measured scattered energy density relative to the incident
  field: < 1e-8.
- **Difference-layer test** — relative field error against TMM ≈ 8e-5 at
  10 elements per wavelength with p = 3, and the error follows the expected
  convergence in both mesh density and polynomial order (measured: 7.2e-4,
  2.3e-4, 1.1e-4, 6.6e-5 at 4, 6, 8, 10 elements per wavelength).

A translation-covariance check confirms that rigidly shifting the mesh on the
periodic torus changes the solution only by the correct Bloch phase.

Tests: `tests/test_solver_m6.py`.

## Layer 5 — local sources

Point, line and sheet current sources are compared against analytic dipole
Green's function solutions in homogeneous media, and multi-source runs are
checked for exact superposition (a combined solve equals the sum of per-source
solves).

Tests: `tests/test_solver_m6b.py`.

## Layer 6 — patterned geometry against independent RCWA

The strongest physics check: real grating patterns are compared against a
rigorous coupled-wave implementation written independently for this purpose
(`tests/reference/rcwa.py`), which shares no code with the FEM path — different
method, different discretization, different author-time.

- 1D line gratings, TE and TM, duty cycles 0.5 and 0.25: diffraction efficiencies
  agree to < 1e-3;
- sloped-sidewall gratings compared against a staircase-sliced RCWA model, where
  the residual difference is the staircasing error of the reference — this is the
  case where the conformal mesh is a genuine advantage;
- 3D patterns (square contact hole, L-shaped feature): energy balance closes to
  ~1e-6 and efficiencies converge under mesh refinement.

Tests: `tests/test_solver_m8.py`, `tests/test_rcwa_selfcheck.py`.

## Layer 7 — outputs

Near-field HDF5 slices, ParaView volumes and diffraction-order tables are
checked for internal consistency: order efficiencies sum to the transmitted plus
reflected flux, Poynting flux through the observation planes balances the input,
and the analytic (0,0) composition matches TMM.

Tests: `tests/test_outputs_m7.py`.

## Layer 8 — GPU paths

The GPU solver and GPU assembly are required to reproduce the CPU result
*exactly*, not merely to converge to the same physics:

| Check | Criterion | Measured |
|---|---|---|
| GPU direct solve vs CPU direct solve (fields, order efficiencies) | < 1e-8 | ≤ 5e-13 |
| GPU vs CPU assembly, element matrices and global CSR | < 1e-13 | ≤ 4e-15 |
| Sparsity structure, GPU vs CPU assembly | bitwise identical | identical |
| Eliminated system and RHS vs the CPU-formed system | < 1e-13 | ≤ 2.4e-15, RHS bitwise equal |
| End-to-end GPU assembly vs CPU assembly (1.34M DOF) | < 1e-10 | 4.2e-12 |
| Diffraction orders, GPU vs CPU assembly (1.34M DOF) | < 1e-8 | 7.0e-13 |
| Analytic TMM through the GPU-assembly path | unchanged from CPU | 7.960e-5 (identical) |

Tests: `tests/test_solver_v2.py`, `tests/test_solver_v2m3.py`,
`tests/test_solver_v2m4.py`, `tests/test_asm_v25.py`.

## Running the suite

```bash
make test          # everything (builds the solver binaries first)
make test-fast     # quick tier
lithofem selftest  # same suite, through the CLI, for verifying an install
```

Tests carrying the `gpu` or `cudss` markers skip automatically when the hardware
or library is absent, so the suite is meaningful on a CPU-only machine.
