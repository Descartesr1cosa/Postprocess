"""Sparse reconstruction operator readers."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from .errors import ValidationError
from .static_sections import read_static_file
from .topology import validate_csr
from .types import Manifest, RankReconstruction, ScalarReconstructionOperator, VectorReconstructionOperator


def read_rank_reconstruction(path: str | Path, *, rank: int, manifest: Manifest) -> RankReconstruction:
    """Read both solver-materialized reconstruction operators."""
    s=read_static_file(path,expected_type="reconstruction",manifest=manifest).sections; v=lambda n:s[n].values
    names={"Bcell_global_id","Bcell_offsets","Bcell_face_ids","Bcell_weights","NodeScalar_global_id","NodeScalar_offsets","NodeScalar_cell_ids","NodeScalar_weights"}
    if set(s)!=names: raise ValidationError(f"{path}: reconstruction section mismatch")
    validate_csr(v("Bcell_offsets"),v("Bcell_face_ids"),name="B_face_to_cell")
    validate_csr(v("NodeScalar_offsets"),v("NodeScalar_cell_ids"),name="cell_scalar_to_node")
    if v("Bcell_offsets").size!=v("Bcell_global_id").size+1 or v("Bcell_weights").shape!=(v("Bcell_face_ids").size,3): raise ValidationError(f"{path}: invalid B reconstruction shape")
    if v("NodeScalar_offsets").size!=v("NodeScalar_global_id").size+1 or v("NodeScalar_weights").size!=v("NodeScalar_cell_ids").size: raise ValidationError(f"{path}: invalid scalar reconstruction shape")
    if not np.all(np.isfinite(v("Bcell_weights"))) or not np.all(np.isfinite(v("NodeScalar_weights"))): raise ValidationError(f"{path}: non-finite reconstruction weights")
    return RankReconstruction(rank,
      VectorReconstructionOperator("B_face_to_cell_cartesian",v("Bcell_global_id"),v("Bcell_offsets"),v("Bcell_face_ids"),v("Bcell_weights")),
      ScalarReconstructionOperator("cell_scalar_to_node",v("NodeScalar_global_id"),v("NodeScalar_offsets"),v("NodeScalar_cell_ids"),v("NodeScalar_weights")))

