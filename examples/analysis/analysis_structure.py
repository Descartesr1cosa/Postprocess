"""Subsolar magnetopause/bow-shock positions and bow-shock jump metrics."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from analysis_core import AnalysisOperation, current_fluid_mask, reconstruct_additive_b_cell


GAMMA = 5.0 / 3.0


def _subsolar_samples(case, state: dict, *, tube_radius_rm: float, min_x_rm: float,
                      max_x_rm: float, neighbours: int) -> dict[str, np.ndarray]:
    """Build locally averaged Cell-centred samples along the subsolar tube."""
    fluid = current_fluid_mask(case)
    xyz = np.asarray(case.cells.coordinates)
    transverse_radius = np.hypot(xyz[:, 1], xyz[:, 2])
    selected = fluid & (transverse_radius <= tube_radius_rm) & (xyz[:, 0] >= min_x_rm) & (xyz[:, 0] <= max_x_rm)
    indices = np.flatnonzero(selected)
    if indices.size < neighbours:
        raise ValueError("Subsolar selection has too few Cells; increase tube radius")
    indices = indices[np.argsort(xyz[indices, 0])]
    fluid_indices = np.flatnonzero(fluid)
    _, local = cKDTree(xyz[fluid]).query(xyz[indices], k=neighbours)
    neighbour_indices = fluid_indices[np.atleast_2d(local)]

    h, na = case.H, case.Na
    density_h = h.number_density("m^-3")
    density_na = na.number_density("m^-3")
    total_density = density_h + density_na
    velocity = (density_h[:, None] * h.velocity_in("m/s") + density_na[:, None] * na.velocity_in("m/s"))
    velocity = np.divide(velocity, total_density[:, None], out=np.full_like(velocity, np.nan), where=total_density[:, None] > 0.0)
    b_total_t = case.unit_converter.convert(
        case.reconstruct_B_cell(case.dynamic_fields) + state["additive_b_cell_nd"],
        quantity="magnetic_field", unit="T",
    )
    current = case.compute_current_dec(unit="A/m^2")
    values = {
        "J_A_m2": np.linalg.norm(current, axis=1),
        "rho_kg_m3": h.mass_density("kg/m^3") + na.mass_density("kg/m^3"),
        "pressure_Pa": h.pressure_in("Pa") + na.pressure_in("Pa"),
        "velocity_m_s": velocity,
        "B_total_T": b_total_t,
        "H_number_density_cm3": h.number_density("cm^-3"),
    }
    return {
        "x_RM": xyz[indices, 0],
        **{name: np.mean(value[neighbour_indices], axis=1) for name, value in values.items()},
    }


def _fast_magnetosonic_mach(samples: dict[str, np.ndarray]) -> np.ndarray:
    """Fast-mode Mach number for propagation along MSO x."""
    rho, pressure, b = (samples[key] for key in ("rho_kg_m3", "pressure_Pa", "B_total_T"))
    mu0 = 4.0e-7 * np.pi
    sound2 = GAMMA * pressure / rho
    b2 = np.einsum("ij,ij->i", b, b)
    alfven2 = b2 / (mu0 * rho)
    cos2 = np.divide(b[:, 0] ** 2, b2, out=np.zeros_like(rho), where=b2 > 0.0)
    discriminant = np.maximum((sound2 + alfven2) ** 2 - 4.0 * sound2 * alfven2 * cos2, 0.0)
    fast2 = 0.5 * (sound2 + alfven2 + np.sqrt(discriminant))
    return np.abs(samples["velocity_m_s"][:, 0]) / np.sqrt(fast2)


def _prepare(case, state: dict) -> None:
    state["additive_b_cell_nd"] = reconstruct_additive_b_cell(case)


def make_global_structure_operation(*, tube_radius_rm: float, min_x_rm: float,
                                  max_x_rm: float, neighbours: int) -> AnalysisOperation:
    """Create the operation for R_MP, R_BS and the resolved bow-shock jump.

    The jump convention is **downstream minus upstream**.  The two sides are
    the locally averaged Cell samples immediately inside/outside the
    fast-magnetosonic Mach-one crossing, so its spatial resolution is explicit
    rather than implying a discontinuity-extrapolated value.
    """
    if neighbours < 1:
        raise ValueError("neighbours must be positive")

    def calculate(case, state: dict) -> dict[str, float]:
        samples = _subsolar_samples(
            case, state, tube_radius_rm=tube_radius_rm, min_x_rm=min_x_rm,
            max_x_rm=max_x_rm, neighbours=neighbours,
        )
        mp_index = int(np.nanargmax(samples["J_A_m2"]))
        mach = _fast_magnetosonic_mach(samples)
        crossings = np.flatnonzero((mach[:-1] <= 1.0) & (mach[1:] > 1.0))
        result = {"R_MP_RM": float(samples["x_RM"][mp_index])}
        if not crossings.size:
            result.update({
                "R_BS_RM": float("nan"),
                "bow_shock_delta_B_nT": float("nan"),
                "bow_shock_delta_total_ion_pressure_nPa": float("nan"),
                "bow_shock_delta_H_number_density_cm3": float("nan"),
            })
            return result
        inner = int(crossings[-1])
        outer = inner + 1
        fraction = (1.0 - mach[inner]) / (mach[outer] - mach[inner])
        result["R_BS_RM"] = float(samples["x_RM"][inner] + fraction * (samples["x_RM"][outer] - samples["x_RM"][inner]))
        result["bow_shock_delta_B_nT"] = float((np.linalg.norm(samples["B_total_T"][inner]) - np.linalg.norm(samples["B_total_T"][outer])) * 1.0e9)
        result["bow_shock_delta_total_ion_pressure_nPa"] = float((samples["pressure_Pa"][inner] - samples["pressure_Pa"][outer]) * 1.0e9)
        result["bow_shock_delta_H_number_density_cm3"] = float(samples["H_number_density_cm3"][inner] - samples["H_number_density_cm3"][outer])
        return result

    return AnalysisOperation(
        name="global_structure",
        calculate=calculate,
        prepare=_prepare,
        configuration={
            "tube_radius_rm": tube_radius_rm,
            "min_x_rm": min_x_rm,
            "max_x_rm": max_x_rm,
            "local_average_cells": neighbours,
            "bow_shock_jump_convention": "downstream_minus_upstream_adjacent_samples",
        },
        units={
            "R_MP_RM": "R_M",
            "R_BS_RM": "R_M",
            "bow_shock_delta_B_nT": "nT",
            "bow_shock_delta_total_ion_pressure_nPa": "nPa",
            "bow_shock_delta_H_number_density_cm3": "cm^-3",
        },
    )
