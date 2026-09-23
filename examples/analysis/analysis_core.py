"""Shared types and physical helpers for modular time-series analysis."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from mpcns_post.assemble import GlobalIDIndex
from mpcns_post.errors import ValidationError


Prepare = Callable[[object, dict], None]
Calculate = Callable[[object, dict], dict[str, float]]


@dataclass(frozen=True)
class AnalysisOperation:
    """One independently switchable analysis operation.

    Put/remove this object on the ``OPERATIONS`` list in ``run_analysis.py``
    (or comment out that single line) to enable/disable the operation.
    """

    name: str
    calculate: Calculate
    units: dict[str, str]
    configuration: dict[str, object] = field(default_factory=dict)
    prepare: Prepare | None = None


def radius_label(radius_rm: float) -> str:
    """Make a stable, Tecplot-column-safe label for a radius in R_M."""
    return f"{float(radius_rm):g}".replace("-", "m").replace(".", "p")


def current_fluid_mask(case) -> np.ndarray:
    """Cells where both ion fluids are defined for coupled measurements."""
    return np.asarray(case.H.valid_mask & case.Na.valid_mask, dtype=bool)


def cell_volume_m3(case) -> np.ndarray:
    """Convert stored nondimensional Cell volumes into m^3."""
    if case.cells.measure is None:
        raise ValueError("Cell volumes are unavailable")
    volume = np.asarray(case.cells.measure, dtype=float)
    if np.any(~np.isfinite(volume)) or np.any(volume <= 0.0):
        raise ValueError("Cell volumes must be finite and positive")
    return volume * float(case.manifest.normalization["length_ref"]) ** 3


def number_inventory(case, number_density_m3: np.ndarray, mask: np.ndarray) -> float:
    """Return the particle inventory over selected Cells."""
    selected = np.asarray(mask, dtype=bool)
    density = np.asarray(number_density_m3, dtype=float)
    if density.shape != selected.shape:
        raise ValueError("number density and inventory mask must have equal shape")
    values = density[selected]
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("selected number density must be finite and non-negative")
    return float(np.sum(values * cell_volume_m3(case)[selected], dtype=np.float64))


def reconstruct_additive_b_cell(case) -> np.ndarray:
    """Reconstruct the prescribed, time-independent Face B field at Cells."""
    face_ids = case.dynamic_fields.global_ids["B_xi"]
    face_index = GlobalIDIndex.build(face_ids)
    face_values = np.zeros(face_ids.size, dtype=np.float64)
    for name in ("Badd_xi", "Badd_eta", "Badd_zeta"):
        chunks = [chunk for chunk in case.rank_constant_fields if name in chunk.fields]
        if not chunks:
            raise KeyError(f"Missing constant field {name}")
        gids = np.concatenate([chunk.global_ids[name] for chunk in chunks])
        values = np.concatenate([np.asarray(chunk.fields[name]).reshape(-1) for chunk in chunks])
        face_values[face_index.lookup(gids)] = values
    operator = case.reconstruction.B_face_to_cell
    local = operator.apply(face_values, face_index)
    output = np.full((case.cells.size, 3), np.nan)
    cell_index = GlobalIDIndex.build(case.cells.global_ids)
    output[cell_index.lookup(operator.output_global_ids)] = local
    if not np.all(np.isfinite(output)):
        raise ValidationError("Additive B reconstruction left unfilled Cells")
    return output
