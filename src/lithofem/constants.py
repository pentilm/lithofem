"""Physical conventions — single source of truth (DESIGN.md §1.3).

All modules must import these; no magic numbers elsewhere.

Conventions:
- Time convention: e^{-i omega t}  =>  lossy media have Im(epsilon) > 0.
- Length unit: nanometre (nm). Angle unit: degree.
- mu_r == 1 everywhere (non-magnetic materials; fixed assumption).
- Right-handed Cartesian (x, y, z); z is the stacking axis.
  Source `from: top` means the source sits on the z_max side and the main
  propagation direction is -z; `from: bottom` is the opposite.
- Plane-wave polarization basis: e_s = (z_hat x k_hat)/|z_hat x k_hat|
  (normal incidence: e_s = y_hat), e_p = k_hat x e_s, so (e_s, e_p, k_hat)
  is right-handed.
"""

from __future__ import annotations

import numpy as np

TIME_CONVENTION = "exp(-i*omega*t)"
LENGTH_UNIT = "nm"
ANGLE_UNIT = "degree"
MU_R = 1.0

# Vacuum impedance in ohms (SI); field/current scalings use this symbolically.
Z0_OHM = 376.730313412


def k0(wavelength_nm: float) -> float:
    """Vacuum wavenumber k0 = 2*pi/lambda in rad/nm."""
    return 2.0 * np.pi / wavelength_nm


def epsilon_from_nk(n: float, k: float) -> complex:
    """Relative permittivity from refractive index: eps = (n + i k)^2.

    With the e^{-i omega t} convention, k >= 0 gives Im(eps) >= 0 (lossy).
    """
    return complex(n, k) ** 2
