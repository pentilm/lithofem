"""Tests for the test harness itself (v2.6).

The suite routes configurations that express no solver preference onto the
GPU. That redirection is only legitimate if it is precise: configurations
that *do* state a preference, and tests that exist to pin the CPU path, must
be left alone. These tests pin that behaviour, so a future change to
conftest cannot silently move the CPU reference tests onto the GPU (which
would leave the fallback path unverified).
"""

from __future__ import annotations

import pytest

from lithofem import config

from . import conftest as ct

REDIRECTED = ct.HAS_GPU and ct.HAS_CUDSS


def _minimal(solver: dict | None = None) -> dict:
    c = {
        "domain": {"Lx": 20.0, "Ly": 20.0, "z_min": 0.0, "z_max": 40.0},
        "materials": {"a": {"n": 1.5, "k": 0.0}},
        "layers": [{"z": [0.0, 20.0], "material": "a"}],
        "frustums": [],
        "wavelength": 13.5,
        "sources": [{"type": "planewave",
                     "incidence": {"theta": 0, "phi": 0, "from": "top"},
                     "polarization": "s"}],
    }
    if solver is not None:
        c["solver"] = solver
    return c


@pytest.mark.fast
@pytest.mark.gpu_ok
@pytest.mark.skipif(not REDIRECTED, reason="no GPU/cuDSS: nothing to redirect")
def test_gpu_ok_test_is_redirected_to_gpu() -> None:
    m = config.expand(_minimal())
    assert m.solver.device == "gpu"
    assert m.solver.type == "direct"
    assert m.fem.assembly == "gpu"


@pytest.mark.fast
def test_explicit_cpu_config_is_left_alone() -> None:
    m = config.expand(_minimal({"type": "direct", "device": "cpu"}))
    assert m.solver.device == "cpu"


@pytest.mark.fast
def test_explicit_iterative_config_is_left_alone() -> None:
    m = config.expand(_minimal({"type": "iterative", "device": "cpu"}))
    assert m.solver.type == "iterative"
    assert m.solver.device == "cpu"


@pytest.mark.fast
def test_unmarked_test_keeps_the_cpu_default() -> None:
    """Routing is opt-in: without `gpu_ok` the config defaults are untouched.

    This is what keeps the tests that assert on default behaviour honest.
    """
    m = config.expand(_minimal())
    assert m.solver.device == "cpu"
    assert m.fem.assembly == "cpu"


@pytest.mark.fast
@pytest.mark.gpu_ok
@pytest.mark.cpu_reference
@pytest.mark.skipif(not REDIRECTED, reason="no GPU/cuDSS")
def test_cpu_reference_wins_over_gpu_ok() -> None:
    """`cpu_reference` pins a test to the CPU even if the module allows GPU."""
    m = config.expand(_minimal())
    assert m.solver.device == "cpu"
