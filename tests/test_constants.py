"""M0 smoke tests: conventions module is importable and self-consistent."""

import numpy as np
import pytest

from lithofem import constants


@pytest.mark.fast
def test_lossy_material_has_positive_im_eps() -> None:
    eps = constants.epsilon_from_nk(0.95, 0.031)
    assert eps.imag > 0.0


@pytest.mark.fast
def test_k0() -> None:
    assert np.isclose(constants.k0(13.5), 2.0 * np.pi / 13.5, rtol=0, atol=1e-15)


@pytest.mark.fast
def test_mu_is_one() -> None:
    assert constants.MU_R == 1.0
