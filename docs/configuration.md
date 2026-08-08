# Configuration reference

A LithoFEM run is fully described by one YAML file. Only `domain`, `wavelength`
and `sources` are strictly required; `materials`, `layers` and `frustums` are the
usual geometry sections and may be empty. Everything else has defaults.

Units: lengths in nanometres, angles in degrees.

The validator reports every problem at once, each with the offending object
index and a suggested fix, before any meshing or solving happens.

## domain (required)

```yaml
domain: {Lx: 96, Ly: 96, z_min: 0, z_max: 120}
```

Lateral unit-cell size and the computational z range, excluding the z-PML that is
appended automatically. `lateral_bc: periodic` is the default and currently the
only accepted value.

## materials / background_epsilon

```yaml
background_epsilon: 1.0            # default medium wherever no layer applies; vacuum by default
materials:
  absorber: {n: 0.95, k: 0.031}    # eps = (n + ik)^2, k >= 0 (lossy => Im eps > 0 under e^{-iwt})
  oxide:    {epsilon: [2.13, 0.0]} # or give [Re eps, Im eps] directly
```

## layers

```yaml
layers:
  - {z: [0, 60], material: absorber}   # z ranges must not overlap; gaps fall back to the background
```

## frustums (extruded polygon features)

```yaml
frustums:
  - vertices: [[24, 24], [72, 24], [72, 72], [24, 72]]  # simple polygon, auto-closed and CCW-normalized
    z0: 0          # base-plane z
    h: 60          # height; may be negative (grows along -z)
    alpha: 85      # sidewall angle to the xy plane, in (0, 180); < 90 tapers away from the base,
                   # > 90 expands; default 90 (vertical)
    epsilon: 1.0   # number, [re, im], or a material name; default vacuum (i.e. an etched hole)
```

Rules enforced by the validator:

- frustums must not overlap each other;
- the mitre offset must remain a single simple polygon over the whole height —
  if it does not, the error reports the maximum `|h|` this frustum allows;
- a frustum may span several layers;
- polygons crossing the periodic cell boundary wrap automatically.

The cross-section convention: the base polygon lives at `z0`, and at height `z`
the cross-section is the base offset by `d(z) = -|z - z0| / tan(alpha)`. This
holds for both signs of `h`, so the solid always tapers (or expands) away from
its base plane.

For sloped *line* features, model the vacuum groove rather than the absorber
line — see the comments in `examples/sloped_grating_tm.yaml` for why.

## wavelength and sources

```yaml
wavelength: 13.5        # single global wavelength shared by all sources
sources:
  - type: planewave
    amplitude: 1.0                             # number or [re, im] (carries a global phase)
    incidence: {theta: 6, phi: 0, from: top}   # from: top | bottom
    polarization: s                            # s | p | {jones: [[re, im], [re, im]]}
  - type: point                                # electric dipole
    position: [48, 48, 90]
    current: [[1, 0], [0, 0], [0, 0]]          # complex vector Z0 * (I*l)
  - type: line
    endpoints: [[10, 10, 90], [80, 10, 90]]
    current: [[0, 0], [1, 0], [0, 0]]
    phase_gradient: 0.0                        # phase gradient along the line, rad/nm
  - type: sheet                                # horizontal (z-normal) rectangle
    corner: [0, 0, 100]
    edges: [[96, 0, 0], [0, 96, 0]]
    current: [[1, 0], [0, 0], [0, 0]]
    phase_gradient: [0.0, 0.0]                 # phase gradients along both edges (directional emission)
```

Semantics:

- Multiple sources in one solve superpose **coherently**. Set
  `output.per_source: true` to additionally write each source's field separately
  (one factorization, reused across right-hand sides).
- A laterally periodic solve carries exactly one Bloch wavevector `k∥`, so plane
  waves with different `k∥` are split automatically into separate solve groups;
  outputs are archived per group as `_g<group>`.
- With local sources only, `k∥` defaults to zero. Set the top-level
  `bloch_k: [kx, ky]` to choose it explicitly (under periodicity, a local source
  is a Bloch-phased array).

## fem / solver

```yaml
fem:
  order: 3                  # Nedelec order, any positive integer
  elems_per_wavelength: 4   # target mesh density (wavelength inside the medium)
  assembly: cpu             # cpu | gpu — GPU matrix assembly (see docs/gpu.md)
  corner_refine: {radius: 20.0, factor: 4.0}   # optional refinement near frustum corners
solver:
  type: direct              # direct (sparse LU) | iterative (GMRES, falls back to direct)
  device: cpu               # cpu | gpu — GPU means the cuDSS direct solver
  gpu_ids: [0]
  gpu_mem_gb: 0             # cuDSS VRAM cap; 0 = automatic (free VRAM)
  rtol: 1e-8                # iterative options
  max_iter: 2000
```

`fem.assembly: gpu` requires `solver.type: direct` and `solver.device: gpu`; if
any GPU step fails, the run logs the reason and falls back to CPU assembly. The
default (`cpu`) leaves the v1 code path untouched.

Solver options exposed on the command line: `--assembly/-a`, `--device`,
`--export-system <prefix>` (write the complex system and RHS in Matrix Market
format, then exit), `--import-solution <file>` (read an externally computed
solution and continue with post-processing), `--mem-limit-gb` (abort if the
CPU factorization memory estimate exceeds the limit),
`--gpu-mem-limit-gb` (cuDSS VRAM cap).

## boundaries.pml

```yaml
boundaries:
  pml: {thickness: 1.0, order: 2, target_reflection: 1e-8}
```

PML on both z faces; `thickness` is measured in wavelengths. The outermost
background material is extended into the PML. Note that the effective absorption
at grazing incidence scales roughly as `target^{cos θ}`, so for θ ≳ 60° a target
of 1e-12 is a better choice.

## sweep (multi-GPU task scheduling)

```yaml
sweep: {gpu_ids: [0, 1], max_parallel: 2}
```

Distributes independent solve groups (for example several incidence angles)
across GPUs as separate processes, one card pinned per task. Failures are
isolated: a failing task does not stop the others, and the summary reports which
group failed and where its log is.

## output

```yaml
output:
  planes:
    - {z: 61.0, quantities: [E, H], resolution: [512, 512], file: nf_z61.h5}
  volume: {enabled: true, file: field_full, include_pml: false}   # ParaView
  orders: {enabled: true}
  per_source: false
```

Observation-plane z values are inserted as mesh breakpoints, so fields are
sampled on mesh faces. `H` is stored as `Z0 * H`, i.e. in the same units as `E`.
The diffraction-order tables are computed on the highest and lowest observation
planes, giving the upward (reflected) and downward (transmitted) directions; the
(0,0) order includes the analytic incident/reflected composition.
