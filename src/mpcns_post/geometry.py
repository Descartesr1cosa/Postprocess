"""Rank-local geometry decoding and validation."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from .errors import ValidationError
from .static_sections import read_static_file
from .types import Manifest, RankGeometry

NAMES=("node_global_id","node_xyz","edge_global_id","edge_node_ids","edge_center_xyz","edge_directed_dr","edge_length","edge_flags","face_global_id","face_center_xyz","face_area_vector","face_area","face_flags","cell_global_id","cell_center_xyz","cell_volume","cell_flags")


def read_rank_geometry(path: str | Path, *, rank: int, manifest: Manifest) -> RankGeometry:
    """Read and validate one owner-only geometry chunk."""
    s=read_static_file(path,expected_type="geometry",manifest=manifest).sections
    missing=set(NAMES)-s.keys()
    if missing or set(s)-set(NAMES): raise ValidationError(f"{path}: geometry section mismatch; missing={sorted(missing)}, extra={sorted(set(s)-set(NAMES))}")
    v=lambda n:s[n].values
    g=RankGeometry(rank,v("node_global_id"),v("node_xyz"),v("edge_global_id"),v("edge_node_ids"),v("edge_center_xyz"),v("edge_directed_dr"),v("edge_length"),v("edge_flags"),v("face_global_id"),v("face_center_xyz"),v("face_area_vector"),v("face_area"),v("face_flags"),v("cell_global_id"),v("cell_center_xyz"),v("cell_volume"),v("cell_flags"))
    groups=((g.node_gid,g.node_xyz,"node"),(g.edge_gid,g.edge_center_xyz,"edge"),(g.face_gid,g.face_center_xyz,"face"),(g.cell_gid,g.cell_center_xyz,"cell"))
    for ids,xyz,name in groups:
        if xyz.shape != (ids.size,3) or not np.all(np.isfinite(xyz)): raise ValidationError(f"{path}: invalid {name} coordinate array")
        if np.unique(ids).size != ids.size: raise ValidationError(f"{path}: duplicate owner {name} gid")
    if g.edge_node_ids.shape!=(g.edge_gid.size,2) or g.edge_dr.shape!=(g.edge_gid.size,3): raise ValidationError(f"{path}: invalid edge array shape")
    collapsed=(g.edge_flags & np.uint32(1<<1))!=0
    calc=np.linalg.norm(g.edge_dr,axis=1)
    if np.any(~collapsed & ((g.edge_length<=0)|~np.isclose(calc,g.edge_length,rtol=1e-10,atol=1e-13))): raise ValidationError(f"{path}: ordinary edge length/dr mismatch")
    fcalc=np.linalg.norm(g.face_area_vector,axis=1)
    if np.any((g.face_area<=0)|~np.isclose(fcalc,g.face_area,rtol=1e-10,atol=1e-13)): raise ValidationError(f"{path}: face area/vector mismatch")
    if np.any(~np.isfinite(g.cell_volume)) or np.any(g.cell_volume<=0): raise ValidationError(f"{path}: invalid cell volume")
    return g

