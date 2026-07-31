"""Shared physics and one-dimensional subsolar sampling helpers."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from mpcns_post.assemble import GlobalIDIndex
from mpcns_post.errors import ValidationError


GAMMA = 5.0 / 3.0


def reconstruct_additive_B_cell(case) -> np.ndarray:
    """Reconstruct the time-independent prescribed B field at cell centres."""
    face_ids = case.dynamic_fields.global_ids["B_xi"]
    face_index = GlobalIDIndex.build(face_ids)
    values = np.zeros(face_ids.size, dtype=np.float64)
    for name in ("Badd_xi", "Badd_eta", "Badd_zeta"):
        chunks = [chunk for chunk in case.rank_constant_fields if name in chunk.fields]
        if not chunks:
            raise KeyError(f"Missing constant field {name}")
        gids = np.concatenate([chunk.global_ids[name] for chunk in chunks])
        field = np.concatenate([np.asarray(chunk.fields[name]).reshape(-1) for chunk in chunks])
        values[face_index.lookup(gids)] = field
    operator = case.reconstruction.B_face_to_cell
    local = operator.apply(values, face_index)
    output = np.full((case.cells.size, 3), np.nan)
    output[GlobalIDIndex.build(case.cells.global_ids).lookup(operator.output_global_ids)] = local
    if not np.all(np.isfinite(output)):
        raise ValidationError("Additive B reconstruction left unfilled cells")
    return output


def prepare_static(case) -> dict:
    """Build geometry-only selection data once for every time sequence."""
    xyz = case.cells.coordinates
    fluid = case.H.valid_mask & case.Na.valid_mask
    return {"xyz_RM": xyz, "fluid_mask": fluid}


def dynamic_quantities(case, additive_B_nd: np.ndarray) -> dict:
    """Calculate only the fields needed by the convergence measurements."""
    fluid = case.H.valid_mask & case.Na.valid_mask
    units = case.unit_converter
    h, na = case.H, case.Na
    rho = h.mass_density("kg/m^3") + na.mass_density("kg/m^3")
    pressure = h.pressure_in("Pa") + na.pressure_in("Pa")
    velocity = (
        h.number_density("m^-3")[:, None] * h.velocity_in("m/s")
        + na.number_density("m^-3")[:, None] * na.velocity_in("m/s")
    ) / (h.number_density("m^-3") + na.number_density("m^-3"))[:, None]
    b_total = units.convert(
        case.reconstruct_B_cell(case.dynamic_fields) + additive_B_nd,
        quantity="magnetic_field", unit="T",
    )
    current = case.compute_current_dec(unit="A/m^2")
    return {
        "fluid_mask": fluid,
        "rho_kg_m3": rho,
        "pressure_Pa": pressure,
        "velocity_m_s": velocity,
        "B_total_T": b_total,
        "J_magnitude_A_m2": np.linalg.norm(current, axis=1),
    }


def make_subsolar_samples(static: dict, quantities: dict, *,
                           transverse_radius_rm: float, min_x_rm: float,
                           max_x_rm: float, neighbours: int) -> dict:
    """Sample an axis tube and smooth each quantity over nearest cells.

    The local 3--5-cell average suppresses isolated wall/current artefacts.
    """
    xyz, fluid = static["xyz_RM"], quantities["fluid_mask"]
    r_perp = np.hypot(xyz[:, 1], xyz[:, 2])
    selected = fluid & (r_perp <= transverse_radius_rm) & (xyz[:, 0] >= min_x_rm) & (xyz[:, 0] <= max_x_rm)
    indices = np.flatnonzero(selected)
    if indices.size < neighbours:
        raise ValueError("Subsolar selection has too few cells; increase SUBSOLAR_TUBE_RADIUS_RM")
    indices = indices[np.argsort(xyz[indices, 0])]
    tree = cKDTree(xyz[fluid])
    _, local = tree.query(xyz[indices], k=neighbours)
    fluid_indices = np.flatnonzero(fluid)
    neighbour_indices = fluid_indices[np.atleast_2d(local)]

    def average(name):
        return np.mean(quantities[name][neighbour_indices], axis=1)

    return {
        "x_RM": xyz[indices, 0],
        "J_A_m2": average("J_magnitude_A_m2"),
        "rho_kg_m3": average("rho_kg_m3"),
        "pressure_Pa": average("pressure_Pa"),
        "velocity_m_s": np.mean(quantities["velocity_m_s"][neighbour_indices], axis=1),
        "B_total_T": np.mean(quantities["B_total_T"][neighbour_indices], axis=1),
    }
