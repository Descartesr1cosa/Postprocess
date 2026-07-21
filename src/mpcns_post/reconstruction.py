"""Sparse reconstruction operator readers."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from .errors import ValidationError
from .static_sections import read_static_file
from .topology import validate_csr
from .types import Manifest, RankReconstruction, ScalarReconstructionOperator, VectorReconstructionOperator


def read_rank_reconstruction(path: str | Path, *, rank: int, manifest: Manifest) -> RankReconstruction:
    """Read v1 reconstruction and optional v3 DEC-current operators."""
    s=read_static_file(path,expected_type="reconstruction",manifest=manifest).sections; v=lambda n:s[n].values
    names={"Bcell_global_id","Bcell_offsets","Bcell_face_ids","Bcell_weights","NodeScalar_global_id","NodeScalar_offsets","NodeScalar_cell_ids","NodeScalar_weights"}
    dec_names={"BfaceJedge_global_id","BfaceJedge_offsets","BfaceJedge_face_ids","BfaceJedge_weights","JedgeJcell_global_id","JedgeJcell_offsets","JedgeJcell_edge_ids","JedgeJcell_weights"}
    missing=names-s.keys(); extra=set(s)-names-dec_names
    if missing or extra or ((set(s)&dec_names) not in (set(),dec_names)):
        raise ValidationError(f"{path}: reconstruction section mismatch; missing={sorted(missing)}, extra={sorted(extra)}")
    validate_csr(v("Bcell_offsets"),v("Bcell_face_ids"),name="B_face_to_cell")
    validate_csr(v("NodeScalar_offsets"),v("NodeScalar_cell_ids"),name="cell_scalar_to_node")
    if v("Bcell_offsets").size!=v("Bcell_global_id").size+1 or v("Bcell_weights").shape!=(v("Bcell_face_ids").size,3): raise ValidationError(f"{path}: invalid B reconstruction shape")
    if v("NodeScalar_offsets").size!=v("NodeScalar_global_id").size+1 or v("NodeScalar_weights").size!=v("NodeScalar_cell_ids").size: raise ValidationError(f"{path}: invalid scalar reconstruction shape")
    if not np.all(np.isfinite(v("Bcell_weights"))) or not np.all(np.isfinite(v("NodeScalar_weights"))): raise ValidationError(f"{path}: non-finite reconstruction weights")
    b_to_j = None
    j_to_cell = None
    if dec_names <= s.keys():
        validate_csr(v("BfaceJedge_offsets"),v("BfaceJedge_face_ids"),name="B_face_to_J_edge")
        validate_csr(v("JedgeJcell_offsets"),v("JedgeJcell_edge_ids"),name="J_edge_to_cell")
        if v("BfaceJedge_offsets").size != v("BfaceJedge_global_id").size + 1:
            raise ValidationError(f"{path}: invalid B_face_to_J_edge row shape")
        if v("BfaceJedge_weights").shape != (v("BfaceJedge_face_ids").size,):
            raise ValidationError(f"{path}: invalid B_face_to_J_edge weights")
        if v("JedgeJcell_offsets").size != v("JedgeJcell_global_id").size + 1:
            raise ValidationError(f"{path}: invalid J_edge_to_cell row shape")
        if v("JedgeJcell_weights").shape != (v("JedgeJcell_edge_ids").size,3):
            raise ValidationError(f"{path}: invalid J_edge_to_cell weights")
        if not np.all(np.isfinite(v("BfaceJedge_weights"))) or not np.all(np.isfinite(v("JedgeJcell_weights"))):
            raise ValidationError(f"{path}: non-finite DEC reconstruction weights")
        b_to_j = ScalarReconstructionOperator(
            "B_face_to_J_edge", v("BfaceJedge_global_id"), v("BfaceJedge_offsets"),
            v("BfaceJedge_face_ids"), v("BfaceJedge_weights"),
        )
        j_to_cell = VectorReconstructionOperator(
            "J_edge_to_cell_cartesian", v("JedgeJcell_global_id"), v("JedgeJcell_offsets"),
            v("JedgeJcell_edge_ids"), v("JedgeJcell_weights"),
        )
    return RankReconstruction(rank,
      VectorReconstructionOperator("B_face_to_cell_cartesian",v("Bcell_global_id"),v("Bcell_offsets"),v("Bcell_face_ids"),v("Bcell_weights")),
      ScalarReconstructionOperator("cell_scalar_to_node",v("NodeScalar_global_id"),v("NodeScalar_offsets"),v("NodeScalar_cell_ids"),v("NodeScalar_weights")),
      b_to_j, j_to_cell)
