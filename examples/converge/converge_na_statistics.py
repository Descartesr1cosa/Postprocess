"""Whole-domain Na+ density statistics and virtual-sphere particle fluxes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class SphereFluxSampler:
    """Static nearest-Cell map for one virtual spherical interface."""

    radius_rm: float
    cell_indices: np.ndarray
    outward_normal: np.ndarray
    sample_area_m2: float


def _fibonacci_sphere(count: int) -> np.ndarray:
    """Return approximately equal-area unit-sphere directions."""
    if count < 8:
        raise ValueError("sphere_sample_count must be at least 8")
    index = np.arange(count, dtype=float)
    z = 1.0 - 2.0 * (index + 0.5) / count
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    azimuth = np.pi * (3.0 - np.sqrt(5.0)) * index
    return np.column_stack((radius * np.cos(azimuth), radius * np.sin(azimuth), z))


def build_sphere_flux_samplers(case, fluid_mask: np.ndarray, radii_rm: tuple[float, ...], *,
                               sphere_sample_count: int) -> tuple[SphereFluxSampler, ...]:
    """Build reusable nearest-fluid-Cell maps for virtual spheres centred at Mercury."""
    radii = tuple(float(radius) for radius in radii_rm)
    if not radii or any(radius <= 0.0 for radius in radii):
        raise ValueError("NA_FLUX_RADII_RM must contain one or more positive radii")
    fluid_indices = np.flatnonzero(np.asarray(fluid_mask, dtype=bool))
    if not fluid_indices.size:
        raise ValueError("No Fluid Cells are available for Na+ sphere-flux sampling")
    normal = _fibonacci_sphere(sphere_sample_count)
    tree = cKDTree(np.asarray(case.cells.coordinates)[fluid_indices])
    length_ref_m = float(case.manifest.normalization["length_ref"])
    samplers = []
    for radius in radii:
        _, local_indices = tree.query(radius * normal, k=1)
        area = 4.0 * np.pi * (radius * length_ref_m) ** 2 / sphere_sample_count
        samplers.append(SphereFluxSampler(
            radius_rm=radius,
            cell_indices=fluid_indices[np.asarray(local_indices, dtype=int)],
            outward_normal=normal,
            sample_area_m2=float(area),
        ))
    return tuple(samplers)


def _radius_label(radius_rm: float) -> str:
    return f"{radius_rm:g}".replace("-", "m").replace(".", "p")


def compute_na_statistics(case, fluid_mask: np.ndarray,
                          samplers: tuple[SphereFluxSampler, ...]) -> dict[str, float]:
    """Return volume-weighted Na+ density and signed outward sphere fluxes.

    Sphere fluxes use equal-area surface samples with nearest Fluid Cell state.
    Positive values are net outward Na+ particle fluxes in particles/s.
    """
    fluid = np.asarray(fluid_mask, dtype=bool)
    na = case.Na
    number_cm3 = na.number_density("cm^-3")
    number_m3 = na.number_density("m^-3")
    velocity_m_s = na.velocity_in("m/s")
    if case.cells.measure is None:
        raise ValueError("Cell volumes are unavailable for Na+ domain statistics")
    volume = np.asarray(case.cells.measure, dtype=float)
    weights = volume[fluid]
    if not weights.size or np.any(weights <= 0.0):
        raise ValueError("Fluid Cell volumes must be positive for Na+ statistics")
    values = number_cm3[fluid]
    mean = float(np.average(values, weights=weights))
    standard_deviation = float(np.sqrt(np.average((values - mean) ** 2, weights=weights)))
    result = {
        "Na_number_density_domain_mean_cm3": mean,
        "Na_number_density_domain_std_cm3": standard_deviation,
    }
    for sampler in samplers:
        indices = sampler.cell_indices
        if not np.all(fluid[indices]):
            raise ValueError("Na+ virtual-sphere sampler reached a currently inactive Fluid Cell")
        radial_flux = number_m3[indices] * np.einsum(
            "ij,ij->i", velocity_m_s[indices], sampler.outward_normal,
        )
        result[f"Na_flux_outward_r{_radius_label(sampler.radius_rm)}_RM_particles_s"] = float(
            np.sum(radial_flux, dtype=float) * sampler.sample_area_m2
        )
    return result
