"""Deterministic random test-stack generator for the TMM module (M1).

All randomness uses a fixed seed; generated cases are reproducible anywhere.
"""

from __future__ import annotations

import numpy as np

SEED = 20260803


def lossless_stacks(n_cases: int = 8, max_layers: int = 8) -> list[dict]:
    """Random lossless stacks (real eps >= 1) with lossless ambients."""
    rng = np.random.default_rng(SEED)
    cases = []
    for _ in range(n_cases):
        m = int(rng.integers(1, max_layers + 1))
        cases.append(
            {
                "eps_in": 1.0 + 0j,
                "eps_out": complex(rng.uniform(1.0, 6.0)),
                "eps": tuple(complex(v) for v in rng.uniform(1.0, 9.0, m)),
                "d": tuple(float(v) for v in rng.uniform(5.0, 120.0, m)),
                "wavelength": float(rng.uniform(150.0, 250.0)),
                "theta_deg": float(rng.uniform(0.0, 80.0)),
            }
        )
    return cases


def complex_stack_10layer() -> dict:
    """One fixed 10-layer complex-eps stack (reciprocity test, M1-2)."""
    rng = np.random.default_rng(SEED + 1)
    m = 10
    eps = rng.uniform(0.7, 6.0, m) + 1j * rng.uniform(0.0, 1.5, m)
    return {
        "eps_in": 1.0 + 0j,
        "eps_out": 1.0 + 0j,
        "eps": tuple(complex(v) for v in eps),
        "d": tuple(float(v) for v in rng.uniform(4.0, 40.0, m)),
        "wavelength": 193.0,
        "theta_deg": 25.0,
    }


def lossy_field_stack() -> dict:
    """Moderately absorbing stack for the internal-field ODE cross-check (M1-4)."""
    return {
        "eps_in": 1.0 + 0j,
        "eps_out": 2.25 + 0j,
        "eps": (2.13 + 0.0j, (0.95 + 0.031j) ** 2, 1.5 + 0.4j, 4.0 + 0.05j),
        "d": (35.0, 50.0, 20.0, 40.0),
        "wavelength": 193.0,
        "theta_deg": 30.0,
    }
