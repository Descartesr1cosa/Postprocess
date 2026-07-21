"""Solver-equivalent DEC reconstruction of current from restart B-face data."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .assemble import GlobalIDIndex
from .errors import ValidationError


@dataclass(frozen=True)
class DECCurrent:
    """Current reconstructed through the solver-materialized DEC operators.

    ``edge_1form`` stores normalized ``J dot dr`` on quotient Edges and
    ``cell_vector`` stores normalized Cartesian current density on Cells.
    """

    edge_global_ids: np.ndarray
    edge_1form: np.ndarray
    cell_global_ids: np.ndarray
    cell_vector: np.ndarray
    debug_edge_max_abs_error: float | None = None


def _extended_B_face_input(case) -> np.ndarray:
    """Build the v3 operator column vector, including raw restart ghosts."""
    if case.latest_restart is None or case.dynamic_fields is None:
        raise ValidationError("load restart data before DEC current reconstruction")
    storages = [topology.B_face_storage for topology in case.rank_topologies]
    if any(storage is None for storage in storages):
        raise ValidationError(
            "DEC current requires v3 DATA_bin Bstore_* topology; regenerate static output"
        )
    concrete = [storage for storage in storages if storage is not None]
    quotient_count = concrete[0].quotient_count
    column_count = concrete[0].global_column_count
    if any(
        storage.quotient_count != quotient_count
        or storage.global_column_count != column_count
        for storage in concrete[1:]
    ):
        raise ValidationError("inconsistent B-face operator column metadata across ranks")

    dynamic = case.dynamic_fields
    face_gids = dynamic.global_ids["B_xi"]
    B_face = sum(np.asarray(dynamic.fields[name]) for name in ("B_xi", "B_eta", "B_zeta"))
    if np.any(face_gids < 0) or np.any(face_gids >= quotient_count):
        raise ValidationError("quotient Face IDs lie outside the v3 DEC column space")
    extended = np.full(column_count, np.nan, dtype=np.float64)
    extended[face_gids] = B_face

    restarts = {restart.rank: restart for restart in case.latest_restart}
    for storage in concrete:
        ghost_rows = np.flatnonzero(storage.quotient_ids < 0)
        for row in ghost_rows:
            rank, block_id, location_code, i, j, k = map(int, storage.addresses[row])
            try:
                restart = restarts[rank]
                field = next(
                    value for value in restart.fields.values()
                    if value.location_code == location_code and value.name.startswith("B_")
                )
                block = field.blocks[block_id]
            except (KeyError, IndexError, StopIteration) as exc:
                raise ValidationError(
                    f"cannot resolve B-face restart storage address {storage.addresses[row].tolist()}"
                ) from exc
            local = (i - block.extent.lo[0], j - block.extent.lo[1], k - block.extent.lo[2])
            if not block.extent.active or any(
                index < 0 or index >= block.values.shape[axis]
                for axis, index in enumerate(local)
            ):
                raise ValidationError(
                    f"B-face restart storage address is outside its active extent: "
                    f"{storage.addresses[row].tolist()}"
                )
            extended[int(storage.global_ids[row])] = block.values[local + (0,)]

    operator = case.reconstruction.B_face_to_J_edge
    assert operator is not None
    needed = np.unique(operator.input_global_ids)
    if np.any(~np.isfinite(extended[needed])):
        bad = needed[~np.isfinite(extended[needed])]
        raise ValidationError(
            f"DEC B-face input has {bad.size} unresolved columns (first={bad[:5].tolist()})"
        )
    return extended


def reconstruct_current(case, *, validate_debug: bool = False) -> DECCurrent:
    """Reconstruct solver-equivalent DEC Edge and Cell current from induced B.

    The static v3 operators already contain the final Hodge, orientation,
    alias-reduction, singular-edge, physical-boundary, and pole corrections.
    Prescribed ``Badd_*`` fields are deliberately excluded, matching Mercury.
    """
    b_to_j = case.reconstruction.B_face_to_J_edge
    j_to_cell = case.reconstruction.J_edge_to_cell
    if b_to_j is None or j_to_cell is None:
        raise ValidationError(
            "DEC current operators are unavailable; this case needs v3 reconstruction_*.bin files"
        )
    extended_B = _extended_B_face_input(case)
    J_edge_local = b_to_j.apply(extended_B)

    edge_ids = case.geometry.edge_gid
    edge_index = GlobalIDIndex.build(edge_ids)
    J_edge = np.full(edge_ids.size, np.nan, dtype=np.float64)
    J_edge[edge_index.lookup(b_to_j.output_global_ids)] = J_edge_local
    if np.any(~np.isfinite(J_edge)):
        raise ValidationError("DEC B_face_to_J_edge left unfilled quotient Edges")

    J_cell_local = j_to_cell.apply(J_edge, edge_index)
    cell_ids = case.geometry.cell_gid
    J_cell = np.full((cell_ids.size, 3), np.nan, dtype=np.float64)
    J_cell[GlobalIDIndex.build(cell_ids).lookup(j_to_cell.output_global_ids)] = J_cell_local
    if np.any(~np.isfinite(J_cell)):
        raise ValidationError("DEC J_edge_to_cell reconstruction left unfilled Cells")

    debug_error = None
    dynamic = case.dynamic_fields
    debug_names = ("J_xi", "J_eta", "J_zeta")
    if dynamic is not None and all(name in dynamic.fields for name in debug_names):
        debug_ids = dynamic.global_ids["J_xi"]
        debug_edge = sum(np.asarray(dynamic.fields[name]) for name in debug_names)
        expected = debug_edge[GlobalIDIndex.build(debug_ids).lookup(edge_ids)]
        debug_error = float(np.max(np.abs(J_edge - expected), initial=0.0))
        if validate_debug:
            scale = float(max(1.0, np.max(np.abs(J_edge), initial=0.0), np.max(np.abs(expected), initial=0.0)))
            tolerance = 2048.0 * np.finfo(np.float64).eps * scale
            if debug_error > tolerance:
                raise ValidationError(
                    f"DEC current does not reproduce debug J_edge: max_abs={debug_error:.17g}, "
                    f"tolerance={tolerance:.17g}"
                )
    elif validate_debug:
        raise ValidationError("validate_debug=True requires optional J_xi/J_eta/J_zeta restart fields")

    return DECCurrent(edge_ids, J_edge, cell_ids, J_cell, debug_error)


def reconstruct_current_cell(case, *, unit: str | None = None, validate_debug: bool = False) -> np.ndarray:
    """Return DEC Cartesian Cell current, normalized or in a requested unit."""
    values = reconstruct_current(case, validate_debug=validate_debug).cell_vector
    if unit is None or unit == "normalized":
        return values
    return case.unit_converter.convert(values, "current_density", unit)


def reconstruct_current_edge(case, *, physical: bool = False, validate_debug: bool = False) -> np.ndarray:
    """Return Edge ``J dot dr``; physical values use A/m."""
    values = reconstruct_current(case, validate_debug=validate_debug).edge_1form
    if not physical:
        return values
    normalization = case.manifest.normalization
    try:
        scale = normalization["current_density_ref"] * normalization["length_ref"]
    except KeyError as exc:
        raise ValidationError("manifest lacks current_density_ref/length_ref") from exc
    return values * scale
