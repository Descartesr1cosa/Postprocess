"""Na+ inventories, boundary budget, asymmetry, and virtual-sphere fluxes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from analysis_core import AnalysisOperation, cell_volume_m3, current_fluid_mask, number_inventory, radius_label


@dataclass(frozen=True)
class SphereFluxSampler:
    """Geometry-only nearest-Cell map for one virtual sphere."""

    radius_rm: float
    cell_indices: np.ndarray
    outward_normal: np.ndarray
    sample_area_m2: float


def _fibonacci_sphere(count: int) -> np.ndarray:
    if count < 8:
        raise ValueError("sphere_sample_count must be at least 8")
    index = np.arange(count, dtype=float)
    z = 1.0 - 2.0 * (index + 0.5) / count
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    phi = np.pi * (3.0 - np.sqrt(5.0)) * index
    return np.column_stack((radius * np.cos(phi), radius * np.sin(phi), z))


def _sphere_flux_names(radius_rm: float) -> dict[str, str]:
    label = radius_label(radius_rm)
    prefix = f"Na_sphere_r{label}_RM"
    return {kind: f"{prefix}_{kind}_particles_s" for kind in ("net", "outward", "inward")}


def make_na_inventory_operation() -> AnalysisOperation:
    """Total physical Na+ particle inventory over all valid Fluid Cells."""
    def calculate(case, state: dict) -> dict[str, float]:
        fluid = current_fluid_mask(case)
        return {"Na_inventory_particles": number_inventory(case, case.Na.number_density("m^-3"), fluid)}

    return AnalysisOperation(
        name="na_inventory", calculate=calculate,
        units={"Na_inventory_particles": "particles"},
    )


def make_na_asymmetry_operation() -> AnalysisOperation:
    """Global y/z inventory splits and A=(N+ - N-)/(N+ + N-)."""
    def calculate(case, state: dict) -> dict[str, float]:
        fluid = current_fluid_mask(case)
        density = case.Na.number_density("m^-3")
        xyz = np.asarray(case.cells.coordinates)
        result = {}
        for axis, title in ((1, "y"), (2, "z")):
            plus = number_inventory(case, density, fluid & (xyz[:, axis] > 0.0))
            minus = number_inventory(case, density, fluid & (xyz[:, axis] < 0.0))
            result[f"Na_inventory_{title}_positive_particles"] = plus
            result[f"Na_inventory_{title}_negative_particles"] = minus
            result[f"Na_asymmetry_{title}_positive_minus_negative"] = (plus - minus) / (plus + minus) if plus + minus > 0.0 else float("nan")
        return result

    units = {}
    for axis in ("y", "z"):
        units[f"Na_inventory_{axis}_positive_particles"] = "particles"
        units[f"Na_inventory_{axis}_negative_particles"] = "particles"
        units[f"Na_asymmetry_{axis}_positive_minus_negative"] = "1"
    return AnalysisOperation(name="na_global_asymmetry", calculate=calculate, units=units)


def make_na_source_operation(*, sodium_load_factor: float) -> AnalysisOperation:
    """Integrate scaled Photo_rate (cm^-3 s^-1) into total Na+ source rate."""
    factor = float(sodium_load_factor)
    if not np.isfinite(factor) or factor < 0.0:
        raise ValueError("sodium_load_factor must be finite and non-negative")

    def calculate(case, state: dict) -> dict[str, float]:
        if factor == 0.0:
            return {"Na_photoionization_source_particles_s": 0.0}
        fluid = current_fluid_mask(case)
        photo_rate_cm3_s = np.asarray(case.get_field("Photo_rate"), dtype=float)
        if photo_rate_cm3_s.shape != fluid.shape or np.any(~np.isfinite(photo_rate_cm3_s[fluid])):
            raise ValueError("Photo_rate must be finite on all Fluid Cells")
        source_particles_s = factor * np.sum(
            photo_rate_cm3_s[fluid] * 1.0e6 * cell_volume_m3(case)[fluid],
            dtype=np.float64,
        )
        return {"Na_photoionization_source_particles_s": float(source_particles_s)}

    return AnalysisOperation(
        name="na_photoionization_source", calculate=calculate,
        units={"Na_photoionization_source_particles_s": "particles/s"},
        configuration={
            "Photo_rate_units": "cm^-3 s^-1",
            "sodium_load_factor": factor,
            "source_definition": "sodium_load_factor * integral(Photo_rate dV)",
        },
    )


def make_na_boundary_budget_operation(*, surface_radius_max_rm: float) -> AnalysisOperation:
    """Split physical-surface loss from outer-boundary escape/inflow.

    Boundary-face orientations are outward from the computational domain.  At
    the Mercury wall this points *toward* the planet, therefore positive flux
    is a surface loss.  At the external boundary positive flux is escape.
    """
    if surface_radius_max_rm <= 0.0:
        raise ValueError("surface_radius_max_rm must be positive")

    def prepare(case, state: dict) -> None:
        radius = lambda xyz, area: np.linalg.norm(xyz, axis=1)
        state["na_surface_boundary"] = case.select_boundary_faces(predicate=lambda xyz, area: radius(xyz, area) <= surface_radius_max_rm)
        state["na_outer_boundary"] = case.select_boundary_faces(predicate=lambda xyz, area: radius(xyz, area) > surface_radius_max_rm)
        if not len(state["na_surface_boundary"]):
            raise ValueError("No exterior faces were classified as the Mercury surface; adjust surface_radius_max_rm")
        if not len(state["na_outer_boundary"]):
            raise ValueError("No exterior faces were classified as outer boundary; adjust surface_radius_max_rm")

    def calculate(case, state: dict) -> dict[str, float]:
        # Some exterior Faces are owned by a Solid/non-applicable Cell.  Na+
        # loss is defined only where the ion state exists, so deliberately
        # omit those owners instead of passing NaNs to the generic all-face API.
        fluid = current_fluid_mask(case)
        density = case.Na.number_density("m^-3")
        velocity = case.Na.velocity_in("m/s")

        def contributions(surface):
            owners = surface.owner_cell_indices
            valid = fluid[owners] & np.isfinite(density[owners]) & np.all(np.isfinite(velocity[owners]), axis=1)
            area_m2 = surface.area_vectors[valid] * float(case.manifest.normalization["length_ref"]) ** 2
            return density[owners[valid]] * np.einsum("ij,ij->i", velocity[owners[valid]], area_m2)

        def split(values):
            positive = float(np.sum(np.maximum(values, 0.0), dtype=np.float64))
            negative = float(-np.sum(np.minimum(values, 0.0), dtype=np.float64))
            return float(np.sum(values, dtype=np.float64)), positive, negative

        surface_net, surface_loss, surface_return = split(contributions(state["na_surface_boundary"]))
        outer_net, outer_escape, outer_inflow = split(contributions(state["na_outer_boundary"]))
        return {
            "Na_surface_net_outward_particles_s": surface_net,
            "Na_surface_loss_particles_s": surface_loss,
            "Na_surface_return_particles_s": surface_return,
            "Na_outer_boundary_net_outward_particles_s": outer_net,
            "Na_outer_boundary_escape_particles_s": outer_escape,
            "Na_outer_boundary_inflow_particles_s": outer_inflow,
        }

    return AnalysisOperation(
        name="na_boundary_budget", calculate=calculate, prepare=prepare,
        configuration={"surface_radius_max_rm": surface_radius_max_rm},
        units={
            "Na_surface_net_outward_particles_s": "particles/s",
            "Na_surface_loss_particles_s": "particles/s",
            "Na_surface_return_particles_s": "particles/s",
            "Na_outer_boundary_net_outward_particles_s": "particles/s",
            "Na_outer_boundary_escape_particles_s": "particles/s",
            "Na_outer_boundary_inflow_particles_s": "particles/s",
        },
    )


def make_na_sphere_flux_operation(*, radii_rm: tuple[float, ...], sphere_sample_count: int) -> AnalysisOperation:
    """Net, outward-only, and inward-only Na+ particle fluxes on spheres."""
    radii = tuple(float(radius) for radius in radii_rm)
    if not radii or any(radius <= 0.0 for radius in radii):
        raise ValueError("radii_rm must contain positive values")

    def prepare(case, state: dict) -> None:
        fluid = current_fluid_mask(case)
        fluid_indices = np.flatnonzero(fluid)
        if not fluid_indices.size:
            raise ValueError("No Fluid Cells are available for sphere-flux sampling")
        normal = _fibonacci_sphere(sphere_sample_count)
        tree = cKDTree(np.asarray(case.cells.coordinates)[fluid_indices])
        length_ref_m = float(case.manifest.normalization["length_ref"])
        samplers = []
        for radius in radii:
            _, local_indices = tree.query(radius * normal, k=1)
            samplers.append(SphereFluxSampler(
                radius, fluid_indices[np.asarray(local_indices, dtype=int)], normal,
                float(4.0 * np.pi * (radius * length_ref_m) ** 2 / sphere_sample_count),
            ))
        state["na_sphere_flux_samplers"] = tuple(samplers)

    def calculate(case, state: dict) -> dict[str, float]:
        fluid = current_fluid_mask(case)
        density = case.Na.number_density("m^-3")
        velocity = case.Na.velocity_in("m/s")
        result = {}
        for sampler in state["na_sphere_flux_samplers"]:
            if not np.all(fluid[sampler.cell_indices]):
                raise ValueError("Virtual-sphere sampler reached an inactive Fluid Cell")
            contribution = density[sampler.cell_indices] * np.einsum("ij,ij->i", velocity[sampler.cell_indices], sampler.outward_normal) * sampler.sample_area_m2
            names = _sphere_flux_names(sampler.radius_rm)
            result[names["net"]] = float(np.sum(contribution, dtype=np.float64))
            result[names["outward"]] = float(np.sum(np.maximum(contribution, 0.0), dtype=np.float64))
            result[names["inward"]] = float(-np.sum(np.minimum(contribution, 0.0), dtype=np.float64))
        return result

    units = {name: "particles/s" for radius in radii for name in _sphere_flux_names(radius).values()}
    return AnalysisOperation(
        name="na_sphere_flux", calculate=calculate, prepare=prepare, units=units,
        configuration={"radii_rm": list(radii), "sphere_sample_count": sphere_sample_count},
    )


def make_na_tail_plane_flux_operation(*, x_rm: float, radius_rm: float, sample_count: int) -> AnalysisOperation:
    """Integrate Na+ transport through a circular cross-tail plane.

    ``u_tail = -u_x`` follows the MSO anti-sunward tail direction.  Equal-area
    samples use the same documented nearest-Fluid-Cell approximation as the
    virtual spheres; positive values are tailward export, not magnetosheath
    escape through the outer computational boundary.
    """
    if radius_rm <= 0.0 or sample_count < 16:
        raise ValueError("tail-plane radius must be positive and sample_count at least 16")

    def prepare(case, state: dict) -> None:
        fluid_indices = np.flatnonzero(current_fluid_mask(case))
        if not fluid_indices.size:
            raise ValueError("No Fluid Cells are available for tail-plane sampling")
        count_axis = int(np.ceil(np.sqrt(sample_count)))
        coordinate = (np.arange(count_axis, dtype=float) + 0.5) / count_axis * 2.0 - 1.0
        yy, zz = np.meshgrid(coordinate * radius_rm, coordinate * radius_rm, indexing="ij")
        disk = yy * yy + zz * zz <= radius_rm * radius_rm
        points = np.column_stack((np.full(np.count_nonzero(disk), x_rm), yy[disk], zz[disk]))
        _, local = cKDTree(np.asarray(case.cells.coordinates)[fluid_indices]).query(points, k=1)
        state["na_tail_plane"] = {
            "indices": fluid_indices[np.asarray(local, dtype=int)],
            "area_m2": float((2.0 * radius_rm / count_axis) ** 2 * float(case.manifest.normalization["length_ref"]) ** 2),
        }

    def calculate(case, state: dict) -> dict[str, float]:
        sampler = state["na_tail_plane"]
        indices = sampler["indices"]
        if not np.all(current_fluid_mask(case)[indices]):
            raise ValueError("Tail-plane sampler reached an inactive Fluid Cell")
        tailward = -case.Na.number_density("m^-3")[indices] * case.Na.velocity_in("m/s")[indices, 0] * sampler["area_m2"]
        return {
            "Na_tail_plane_net_tailward_particles_s": float(np.sum(tailward, dtype=np.float64)),
            "Na_tail_plane_tailward_outflow_particles_s": float(np.sum(np.maximum(tailward, 0.0), dtype=np.float64)),
            "Na_tail_plane_sunward_inflow_particles_s": float(-np.sum(np.minimum(tailward, 0.0), dtype=np.float64)),
        }

    return AnalysisOperation(
        name="na_tail_plane_flux", calculate=calculate, prepare=prepare,
        configuration={"x_rm": x_rm, "radius_rm": radius_rm, "requested_sample_count": sample_count, "direction": "tailward=-x"},
        units={
            "Na_tail_plane_net_tailward_particles_s": "particles/s",
            "Na_tail_plane_tailward_outflow_particles_s": "particles/s",
            "Na_tail_plane_sunward_inflow_particles_s": "particles/s",
        },
    )
