"""Per-time complete 3-D Fluid Node Tecplot exports for analysis fields."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mpcns_post.tecplot import (
    TecplotZone,
    inspect_tecplot_binary,
    project_cell_values_to_nodes,
    write_tecplot_binary,
)

from analysis_core import AnalysisOperation, current_fluid_mask, reconstruct_additive_b_cell


def _cell_fields(case, state: dict) -> dict[str, np.ndarray]:
    """Build all requested diagnostics at Cell centres before Node projection."""
    h, na = case.H, case.Na
    n_h = h.number_density("cm^-3")
    n_na = na.number_density("cm^-3")
    n_total = n_h + n_na
    n_h_m3 = h.number_density("m^-3")
    n_na_m3 = na.number_density("m^-3")
    n_total_m3 = n_h_m3 + n_na_m3
    bulk = n_h_m3[:, None] * h.velocity_in("km/s") + n_na_m3[:, None] * na.velocity_in("km/s")
    bulk = np.divide(bulk, n_total_m3[:, None], out=np.full_like(bulk, np.nan), where=n_total_m3[:, None] > 0.0)
    pressure_h = h.pressure_in("nPa")
    pressure_na = na.pressure_in("nPa")
    b_nt = case.unit_converter.convert(
        case.reconstruct_B_cell(case.dynamic_fields) + state["additive_b_cell_nd"],
        quantity="magnetic_field", unit="nT",
    )
    current = case.compute_current_dec(unit="nA/m^2")
    b_magnitude = np.linalg.norm(b_nt, axis=1)
    b_hat = np.divide(b_nt, b_magnitude[:, None], out=np.zeros_like(b_nt), where=b_magnitude[:, None] > 0.0)
    j_parallel = np.einsum("ij,ij->i", current, b_hat)
    j_magnitude = np.linalg.norm(current, axis=1)
    j_perpendicular = np.sqrt(np.maximum(j_magnitude * j_magnitude - j_parallel * j_parallel, 0.0))
    mass_na = na.mass_density("kg/m^3")
    mass_total = mass_na + h.mass_density("kg/m^3")
    pressure_total = pressure_h + pressure_na
    return {
        "U_x_km_s": bulk[:, 0], "U_y_km_s": bulk[:, 1], "U_z_km_s": bulk[:, 2],
        "n_H_cm3": n_h, "n_Na_cm3": n_na, "n_total_cm3": n_total,
        "p_H_nPa": pressure_h, "p_Na_nPa": pressure_na, "p_total_nPa": pressure_total,
        "B_x_nT": b_nt[:, 0], "B_y_nT": b_nt[:, 1], "B_z_nT": b_nt[:, 2],
        "J_x_nA_m2": current[:, 0], "J_y_nA_m2": current[:, 1], "J_z_nA_m2": current[:, 2],
        "J_magnitude_nA_m2": j_magnitude,
        "J_perpendicular_nA_m2": j_perpendicular,
        "FAC_nA_m2": j_parallel,
        "Na_to_H_number_ratio": np.divide(n_na, n_h, out=np.full_like(n_na, np.nan), where=n_h > 0.0),
        "Na_mass_fraction": np.divide(mass_na, mass_total, out=np.full_like(mass_na, np.nan), where=mass_total > 0.0),
        "Na_pressure_fraction": np.divide(pressure_na, pressure_total, out=np.full_like(pressure_na, np.nan), where=pressure_total > 0.0),
    }


def _node_fields(case, cell_fields: dict[str, np.ndarray], fluid: np.ndarray) -> dict[str, np.ndarray]:
    """Project all diagnostics once to globally shared Fluid Nodes."""
    names = tuple(cell_fields)
    projected = project_cell_values_to_nodes(
        case,
        np.column_stack([cell_fields[name] for name in names]),
        valid_mask=fluid,
    )
    return {
        "X": case.nodes.coordinates[:, 0],
        "Y": case.nodes.coordinates[:, 1],
        "Z": case.nodes.coordinates[:, 2],
        **{name: projected[:, index] for index, name in enumerate(names)},
    }


def _fluid_node_zones(case, values: dict[str, np.ndarray]) -> list[TecplotZone]:
    """Return full ordered Node blocks, sharing values at every interface."""
    node_blocks = {(block.rank, block.block_id): block for block in case.iter_blocks("node")}
    zones = []
    for cell_block in case.iter_blocks("cell"):
        if cell_block.physics != "Fluid":
            continue
        try:
            node_block = node_blocks[(cell_block.rank, cell_block.block_id)]
        except KeyError as exc:
            raise ValueError(f"Fluid block {cell_block.rank}/{cell_block.block_id} has no Node map") from exc
        zone_values = {name: node_block.reshape(array) for name, array in values.items()}
        for name, array in zone_values.items():
            if not np.all(np.isfinite(array)):
                raise ValueError(
                    f"rank {cell_block.rank} block {cell_block.block_id} has non-finite Fluid Node field {name}"
                )
        zones.append(TecplotZone(
            f"rank{cell_block.rank:04d}_block{cell_block.block_id:04d}_Fluid",
            "Fluid",
            zone_values,
        ))
    if not zones:
        raise ValueError("No Fluid Node blocks are available for 3-D Tecplot export")
    return zones


def make_node_volume_operation(*, output_dir: Path) -> AnalysisOperation:
    """Write one connected 3-D Fluid Node ``<time>.plt`` file per time."""
    destination = Path(output_dir)

    def prepare(case, state: dict) -> None:
        if "additive_b_cell_nd" not in state:
            state["additive_b_cell_nd"] = reconstruct_additive_b_cell(case)
        destination.mkdir(parents=True, exist_ok=True)

    def calculate(case, state: dict) -> dict[str, float]:
        fluid = current_fluid_mask(case)
        values = _node_fields(case, _cell_fields(case, state), fluid)
        time = float(case.latest_restart[0].time)
        path = destination / f"{time:.12e}.plt"
        info = inspect_tecplot_binary(write_tecplot_binary(
            path,
            title="MPCNS 3-D time-series analysis fields at globally shared Fluid Nodes",
            variable_names=tuple(values),
            zones=_fluid_node_zones(case, values),
            solution_time=time,
        ))
        print(f"  3-D Tecplot: {info.path.name} ({len(info.zones)} Fluid zones)")
        return {"node_volume_plt_written": 1.0}

    return AnalysisOperation(
        name="node_volume_tecplot", calculate=calculate, prepare=prepare,
        configuration={"output_dir": str(destination), "location": "node", "geometry": "full_3d_fluid_blocks"},
        units={"node_volume_plt_written": "1"},
    )
