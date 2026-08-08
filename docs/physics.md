# Formulation and conventions

## Conventions

| Quantity | Convention |
|---|---|
| Time factor | `e^{-iωt}`, so lossy media have `Im ε > 0` and `ε = (n + ik)²` with `k ≥ 0` |
| Length | nanometres |
| Angles | degrees |
| Permeability | `μ ≡ 1` (non-magnetic) |
| Stacking axis | `z`; `from: top` means the source sits on the `z_max` side and propagates along `−z` |
| s polarization | `ê_s = (ẑ × k̂)/|ẑ × k̂|`, so `ê_s = ŷ` at normal incidence |
| p polarization | `ê_p = k̂ × ê_s` |
| Magnetic field | stored as `Z₀·H`, i.e. in the same units as `E` |

These are collected in `src/lithofem/constants.py` so that every module — the
transfer-matrix reference, the mesher, the solver and the post-processing —
shares one definition.

## Governing equation

LithoFEM solves the time-harmonic vector wave equation for the electric field on
a periodic unit cell:

```
∇ × (μ⁻¹ ∇ × E) − k₀² ε E = 0,     k₀ = 2π/λ
```

with a z-direction perfectly matched layer, Bloch-periodic lateral boundaries,
and PEC caps outside the PML.

## Scattered-field formulation

Rather than solving for the total field, LithoFEM splits

```
E_total = E_inc + E_sc
```

where `E_inc` is the **analytic** field of the unpatterned layer stack, computed
by the built-in transfer-matrix (TMM) module — evaluated exactly, not
discretized. Substituting gives an equation for the scattered field driven by
the contrast between the actual permittivity and the background stack:

```
∇ × (∇ × E_sc) − k₀² ε E_sc = k₀² (ε − ε_bg) E_inc
```

The right-hand side is non-zero only inside patterned (frustum) regions. Two
consequences matter in practice:

- an unpatterned stack produces exactly zero scattered field, which makes the
  formulation self-testing (this is regression test M6-1);
- the incident field never suffers discretization error, so mesh resolution is
  spent only on the scattered part.

## Bloch periodicity via an envelope substitution

Under lateral periodicity with Bloch wavevector `k∥ = (kx, ky)`, the scattered
field satisfies `E_sc(r + a) = E_sc(r) e^{i k∥·a}`. Instead of imposing phase-
shifted constraints between periodic mesh faces, LithoFEM substitutes

```
E_sc = u(r) e^{i k∥·r}
```

and solves for the **envelope** `u`, which is strictly periodic. Periodic mesh
faces can then be merged with plain identity constraints, and the `k∥`
dependence moves into the bilinear form as additional `i k∥ ×` coupling terms:
two mixed curl operators and one mass-like term.

A practical consequence worth knowing: at oblique incidence the resulting system
matrix is **not** complex symmetric (the mixed terms change sign under
transposition), so any solver option that assumes symmetry must not be used —
LithoFEM always configures its direct solvers for general matrices.

Since a single solve carries one `k∥`, plane waves with different incidence
angles are automatically split into separate solve groups.

## z-PML

Absorption along `z` uses a complex coordinate stretch `z → z̃` with stretch
factor `s(z)`, implemented as an anisotropic material tensor:

```
Λ = diag(s, s, 1/s),   with   ∇ × (Λ⁻¹ ∇ × E) − k₀² ε Λ E = 0
```

The profile is polynomial, `s(z) = 1 + i·σ_max·d^m`, where `d ∈ [0,1]` is the
normalized depth into the PML and `σ_max` is derived from the requested target
reflection, the polynomial order, and the refractive index of the medium being
extended into the layer.

The material of the outermost background layer is continued into the PML, so no
impedance mismatch is introduced at the interface.

Note that the effective absorption degrades at grazing incidence roughly as
`target^{cos θ}`. For θ ≳ 60°, request `target_reflection: 1e-12` rather than the
default 1e-8.

## Discretization

Nédélec H(curl) elements of arbitrary order on conformal tetrahedra. Tangential
continuity across faces is built into the space, which is what makes the
discrete curl-curl operator well behaved; normal components are free to jump at
material interfaces, as physics requires.

The complex system is assembled as a sesquilinear form (real and imaginary parts
kept as two real blocks with identical sparsity) and handed to a complex sparse
direct solver.

Because the geometry is meshed conformally, sloped and non-Manhattan sidewalls
are represented by the element faces themselves — there is no staircase
approximation of the kind that RCWA (slicing) and FDTD (voxels) must make.

## Diffraction orders

On the highest and lowest observation planes, the field envelope is projected
onto the propagating and evanescent Rayleigh orders of the surrounding medium.
For each order the output carries complex s/p amplitudes, the propagation
direction, the z-flux, and a propagating/evanescent flag. The (0,0) order
combines the analytic incident and reflected contributions with the computed
scattered field.

Summing the z-flux over all propagating orders and comparing against the input
gives an energy-balance check that is part of the test suite.
