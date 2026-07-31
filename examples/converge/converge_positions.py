"""Subsolar magnetopause and bow-shock location measurements."""

from __future__ import annotations

import numpy as np

from converge_core import GAMMA


def magnetopause_x(samples: dict) -> float:
    """Define the subsolar magnetopause as the smoothed |J| maximum."""
    return float(samples["x_RM"][np.nanargmax(samples["J_A_m2"])])


def fast_magnetosonic_mach(samples: dict) -> np.ndarray:
    """Fast-mode Mach number for propagation along the MSO x direction."""
    rho, pressure, b = (samples[key] for key in ("rho_kg_m3", "pressure_Pa", "B_total_T"))
    mu0 = 4.0e-7 * np.pi
    sound2 = GAMMA * pressure / rho
    alfven2 = np.sum(b * b, axis=1) / (mu0 * rho)
    cos2 = np.divide(b[:, 0] ** 2, np.sum(b * b, axis=1), out=np.zeros_like(rho), where=np.sum(b * b, axis=1) > 0.0)
    fast2 = 0.5 * (sound2 + alfven2 + np.sqrt(np.maximum((sound2 + alfven2) ** 2 - 4.0 * sound2 * alfven2 * cos2, 0.0)))
    return np.abs(samples["velocity_m_s"][:, 0]) / np.sqrt(fast2)


def bow_shock_x(samples: dict) -> float:
    """Find outermost M_f=1 crossing (super-fast upstream to sub-fast inside)."""
    x = samples["x_RM"]
    mach = fast_magnetosonic_mach(samples)
    crossings = np.flatnonzero((mach[:-1] <= 1.0) & (mach[1:] > 1.0))
    if not crossings.size:
        return float("nan")
    index = int(crossings[-1])
    fraction = (1.0 - mach[index]) / (mach[index + 1] - mach[index])
    return float(x[index] + fraction * (x[index + 1] - x[index]))
