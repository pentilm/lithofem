# Contributing

Thanks for your interest in LithoFEM. Issues and pull requests are welcome.

## Before opening a pull request

```bash
pip install -e ".[dev]"
make solver           # C++ binaries (needs an MFEM build, see docs/building.md)
make test             # full suite; make test-fast for the quick tier
ruff check src tests tools
mypy src
```

Tests marked `gpu` or `cudss` skip automatically without the corresponding
hardware or library, so a CPU-only machine can still run a meaningful suite.

## What a change should come with

LithoFEM's value rests on its numerical results being checkable, so new physics
or numerics must arrive with a check against something independent — an analytic
solution, a manufactured solution, an established reference implementation, or
an exact equivalence to an already-verified code path. `docs/validation.md`
describes how the existing layers are pinned down; please extend that chain
rather than adding an untested branch beside it.

Numerical criteria in tests should be hard thresholds with a stated origin
(machine precision, discretization error, a documented tolerance), not values
tuned until the test passes.

## Conventions

- Physics conventions (time factor, units, polarization basis) live in
  `src/lithofem/constants.py` — use them rather than re-deriving constants.
- The YAML schema only grows: new keys need defaults that preserve existing
  behaviour exactly.
- CUDA kernels must stay architecture-neutral (standard CUDA C++, no
  `__CUDA_ARCH__` branches, no architecture-specific instructions), so the code
  remains portable across GPU generations.
- Python is type-annotated and checked with mypy; C++ follows the surrounding
  MFEM-style formatting.

## Reporting problems

For numerical issues, please include the YAML configuration, the LithoFEM
version, the MFEM configuration (`config/config.mk` flags), GPU model and driver
version if relevant, and the `run_log.jsonl` from the failing run.
