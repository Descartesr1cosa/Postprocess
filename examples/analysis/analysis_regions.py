"""Configurable geometric Na+ regions and their enclosing-boundary fluxes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mpcns_post.assemble import GlobalIDIndex

from analysis_core import AnalysisOperation, cell_volume_m3, current_fluid_mask, number_inventory


@dataclass(frozen=True)
class RegionTopology:
    """One static, compact face-to-Cell map reused by every region/time."""

    first_cell_indices: np.ndarray
    second_cell_indices: np.ndarray
    first_signs: np.ndarray
    second_signs: np.ndarray
    face_indices: np.ndarray


def _row_maps(relation) -> dict[int, tuple[np.ndarray, np.ndarray | None]]:
    if relation.row_global_ids is None:
        raise ValueError("Topology relation has no global row IDs")
    rows = {}
    for index, gid in enumerate(relation.row_global_ids):
        start, stop = relation.offsets[index:index + 2]
        rows[int(gid)] = (
            relation.indices[start:stop],
            None if relation.signs is None else relation.signs[start:stop],
        )
    return rows


def _mask_from_spec(xyz: np.ndarray, spec: dict) -> np.ndarray:
    """Select a rectangular/radial region from a serializable specification."""
    mask = np.ones(xyz.shape[0], dtype=bool)
    for axis, component in (("x", 0), ("y", 1), ("z", 2)):
        interval = spec.get(axis)
        if interval is not None:
            if len(interval) != 2 or float(interval[0]) > float(interval[1]):
                raise ValueError(f"Region {spec['name']!r} has invalid {axis} interval")
            mask &= (xyz[:, component] >= float(interval[0])) & (xyz[:, component] <= float(interval[1]))
    radius = np.linalg.norm(xyz, axis=1)
    if "r" in spec:
        lower, upper = map(float, spec["r"])
        if lower < 0.0 or lower > upper:
            raise ValueError(f"Region {spec['name']!r} has invalid r interval")
        mask &= (radius >= lower) & (radius <= upper)
    return mask


def _build_region_topology(case) -> RegionTopology:
    """Build a one-time manifold Face/Cell incidence map.

    The old direct approach scanned all Faces once *per region*.  This map
    scans them once and lets each subsequent region apply vectorized masks.
    It keeps only two Cell indices and two orientation signs per Face.
    """
    cell_index = GlobalIDIndex.build(case.cells.global_ids)
    face_index = GlobalIDIndex.build(case.faces.global_ids)
    face_rows = _row_maps(case.topology.face_to_cell)
    cell_rows = _row_maps(case.topology.cell_to_face)
    face_gids = np.fromiter(face_rows, dtype=np.int64, count=len(face_rows))
    first_gid = np.full(face_gids.size, -1, dtype=np.int64)
    second_gid = np.full(face_gids.size, -1, dtype=np.int64)
    first_sign = np.zeros(face_gids.size, dtype=np.int8)
    second_sign = np.zeros(face_gids.size, dtype=np.int8)
    for position, face_gid in enumerate(face_gids):
        neighbour_gids, _ = face_rows[int(face_gid)]
        if neighbour_gids.size not in (1, 2):
            raise ValueError(f"Face {face_gid} has {neighbour_gids.size} incident Cells; expected a manifold mesh")
        for slot, cell_gid in enumerate(neighbour_gids):
            cell_gid = int(cell_gid)
            face_ids, signs = cell_rows[cell_gid]
            if signs is None:
                raise ValueError("cell_to_face orientation signs are unavailable")
            match = np.flatnonzero(face_ids == face_gid)
            if match.size != 1:
                raise ValueError(f"Missing/duplicate Face {face_gid} on Cell {cell_gid}")
            if slot == 0:
                first_gid[position] = cell_gid
                first_sign[position] = int(signs[match[0]])
            else:
                second_gid[position] = cell_gid
                second_sign[position] = int(signs[match[0]])
    first = cell_index.lookup(first_gid).astype(np.int32, copy=False)
    second = np.full(second_gid.shape, -1, dtype=np.int32)
    has_second = second_gid >= 0
    second[has_second] = cell_index.lookup(second_gid[has_second]).astype(np.int32, copy=False)
    return RegionTopology(
        first, second, first_sign, second_sign,
        face_index.lookup(face_gids).astype(np.int32, copy=False),
    )


def _boundary_flux(case, topology: RegionTopology, inside: np.ndarray,
                   density: np.ndarray, velocity: np.ndarray) -> np.ndarray:
    """Return per-Face fluxes, oriented outward from one selected region."""
    first = topology.first_cell_indices
    second = topology.second_cell_indices
    first_inside = inside[first]
    second_valid = second >= 0
    second_inside = np.zeros(second.shape, dtype=bool)
    second_inside[second_valid] = inside[second[second_valid]]
    boundary = first_inside ^ second_inside
    use_first = boundary & first_inside
    use_second = boundary & second_inside
    owner = np.where(use_first, first, second)
    signs = np.where(use_first, topology.first_signs, topology.second_signs)
    area = case.faces.area_vectors[topology.face_indices[boundary]] * signs[boundary, None]
    owner = owner[boundary]
    return density[owner] * np.einsum(
        "ij,ij->i", velocity[owner], area * float(case.manifest.normalization["length_ref"]) ** 2,
    )


def make_na_region_operation(*, regions: tuple[dict, ...]) -> AnalysisOperation:
    """Return inventories and net/outward/inward enclosing flux per region.

    Region definitions are intentionally geometric and explicit: ``x``, ``y``,
    ``z``, and ``r`` each accept a two-value interval in R_M.  They are useful
    reproducible proxies for cusp/mantle/sheet/tail attribution; tune the
    bounds in the runner to the particular simulation geometry.
    """
    definitions = tuple(dict(item) for item in regions)
    names = [str(item.get("name", "")) for item in definitions]
    if not definitions or any(not name.replace("_", "").isalnum() for name in names) or len(set(names)) != len(names):
        raise ValueError("Regions need unique alphanumeric/underscore names")

    def prepare(case, state: dict) -> None:
        fluid = current_fluid_mask(case)
        xyz = np.asarray(case.cells.coordinates)
        prepared = {}
        for spec in definitions:
            name = str(spec["name"])
            mask = _mask_from_spec(xyz, spec) & fluid
            prepared[name] = mask
        state["na_regions"] = prepared
        state["na_region_xyz"] = xyz
        state["na_region_topology"] = _build_region_topology(case)

    def calculate(case, state: dict) -> dict[str, float]:
        density = case.Na.number_density("m^-3")
        velocity = case.Na.velocity_in("m/s")
        particle_cells = density * cell_volume_m3(case)
        xyz = state["na_region_xyz"]
        result = {}
        topology = state["na_region_topology"]
        for name, mask in state["na_regions"].items():
            prefix = f"Na_region_{name}"
            result[f"{prefix}_cell_count"] = float(np.count_nonzero(mask))
            if not np.any(mask):
                result[f"{prefix}_inventory_particles"] = float("nan")
                result[f"{prefix}_flux_net_particles_s"] = float("nan")
                result[f"{prefix}_flux_outward_particles_s"] = float("nan")
                result[f"{prefix}_flux_inward_particles_s"] = float("nan")
                for axis in ("y", "z"):
                    result[f"{prefix}_asymmetry_{axis}_positive_minus_negative"] = float("nan")
                continue
            result[f"{prefix}_inventory_particles"] = number_inventory(case, density, mask)
            contribution = _boundary_flux(case, topology, mask, density, velocity)
            result[f"{prefix}_flux_net_particles_s"] = float(np.sum(contribution, dtype=np.float64))
            result[f"{prefix}_flux_outward_particles_s"] = float(np.sum(np.maximum(contribution, 0.0), dtype=np.float64))
            result[f"{prefix}_flux_inward_particles_s"] = float(-np.sum(np.minimum(contribution, 0.0), dtype=np.float64))
            for component, axis in ((1, "y"), (2, "z")):
                plus = float(np.sum(particle_cells[mask & (xyz[:, component] > 0.0)], dtype=np.float64))
                minus = float(np.sum(particle_cells[mask & (xyz[:, component] < 0.0)], dtype=np.float64))
                result[f"{prefix}_asymmetry_{axis}_positive_minus_negative"] = (plus - minus) / (plus + minus) if plus + minus > 0.0 else float("nan")
        return result

    units = {}
    for name in names:
        prefix = f"Na_region_{name}"
        units[f"{prefix}_cell_count"] = "cells"
        units[f"{prefix}_inventory_particles"] = "particles"
        units[f"{prefix}_flux_net_particles_s"] = "particles/s"
        units[f"{prefix}_flux_outward_particles_s"] = "particles/s"
        units[f"{prefix}_flux_inward_particles_s"] = "particles/s"
        units[f"{prefix}_asymmetry_y_positive_minus_negative"] = "1"
        units[f"{prefix}_asymmetry_z_positive_minus_negative"] = "1"
    return AnalysisOperation(
        name="na_regions", calculate=calculate, prepare=prepare, units=units,
        configuration={"regions": definitions, "flux_method": "Cell-owned values on topology-defined enclosing Faces"},
    )
