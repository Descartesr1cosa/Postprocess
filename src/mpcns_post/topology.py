"""CSR and structured local-map topology readers."""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np
from .errors import ValidationError
from .static_sections import read_static_file
from .types import CSRConnectivity, LocalEntityMap, Manifest, RankTopology

LOCATIONS=("node","cell","edge_xi","edge_eta","edge_zeta","face_xi","face_eta","face_zeta")
MAP_RE=re.compile(r"^(b\d{4}_\d+)_(shape|gid|sign|owner)$")


def validate_csr(offsets,indices,*,signs=None,name="CSR") -> None:
    """Validate the structural invariants of a CSR relation."""
    offsets=np.asarray(offsets); indices=np.asarray(indices)
    if offsets.ndim!=1 or offsets.size==0 or offsets[0]!=0 or np.any(np.diff(offsets)<0) or offsets[-1]!=indices.size: raise ValidationError(f"{name}: invalid CSR offsets")
    if indices.ndim!=1 or np.any(indices<0): raise ValidationError(f"{name}: invalid CSR indices")
    if signs is not None:
        signs=np.asarray(signs)
        if signs.shape!=indices.shape or np.any((signs!=-1)&(signs!=1)): raise ValidationError(f"{name}: signs must align and equal +/-1")


def read_rank_topology(path: str | Path, *, rank: int, manifest: Manifest) -> RankTopology:
    """Read owner incidence relations plus rank/block structured maps."""
    s=read_static_file(path,expected_type="topology",manifest=manifest).sections; v=lambda n:s[n].values
    required={"face_global_id","face_edge_offsets","face_edge_ids","face_edge_signs","cell_global_id","cell_face_offsets","cell_face_ids","cell_face_signs",
      "node_global_id","node_cell_offsets","node_cell_ids","edge_global_id","edge_cell_global_id","edge_cell_offsets","edge_cell_ids","face_cell_global_id","face_cell_offsets","face_cell_ids","block_connections","edge_node_ids"}
    missing=required-s.keys()
    if missing: raise ValidationError(f"{path}: missing topology sections {sorted(missing)}")
    f2e=CSRConnectivity(v("face_edge_offsets"),v("face_edge_ids"),v("face_edge_signs").astype(np.int32)); validate_csr(f2e.offsets,f2e.indices,signs=f2e.signs,name="face_to_edge")
    c2f=CSRConnectivity(
        v("cell_face_offsets"),
        v("cell_face_ids"),
        v("cell_face_signs").astype(np.int32),
        row_global_ids=v("cell_global_id"),
    )
    validate_csr(c2f.offsets,c2f.indices,signs=c2f.signs,name="cell_to_face")
    if c2f.row_global_ids.size != c2f.offsets.size - 1:
        raise ValidationError(f"{path}: cell_to_face row key mismatch")
    rel=[]
    for prefix in ("node","edge","face"):
        key_name=prefix+"_global_id" if prefix=="node" else prefix+"_cell_global_id"
        x=CSRConnectivity(v(prefix+"_cell_offsets"),v(prefix+"_cell_ids"),row_global_ids=v(key_name)); validate_csr(x.offsets,x.indices,name=prefix+"_to_cell")
        if x.row_global_ids.size != x.offsets.size-1: raise ValidationError(f"{path}: {prefix} reverse-row key mismatch")
        rel.append(x)
    bases={}
    for name in s:
        m=MAP_RE.match(name)
        if m: bases.setdefault(m.group(1),set()).add(m.group(2))
    maps=[]
    for base,parts in sorted(bases.items()):
        if parts!={"shape","gid","sign","owner"}: raise ValidationError(f"{path}: incomplete local map {base}")
        sh=np.asarray(v(base+"_shape")).reshape(-1)
        if sh.size!=5: raise ValidationError(f"{path}: invalid local map shape {base}")
        block,loc,ni,nj,nk=map(int,sh); size=ni*nj*nk
        if loc not in range(8) or min(ni,nj,nk)<0: raise ValidationError(f"{path}: invalid local map metadata {base}")
        gids=v(base+"_gid"); signs=v(base+"_sign").astype(np.int32); owners=v(base+"_owner").astype(bool)
        if any(x.size!=size for x in (gids,signs,owners)) or np.any((v(base+"_owner")!=0)&(v(base+"_owner")!=1)): raise ValidationError(f"{path}: invalid local map sizes/owners {base}")
        allowed=(signs==1) if loc in (0,1) else ((signs==1)|(signs==-1))
        if not np.all(allowed): raise ValidationError(f"{path}: invalid orientation signs {base}")
        maps.append(LocalEntityMap(block,LOCATIONS[loc],(ni,nj,nk),gids,signs,owners))
    bc=v("block_connections")
    return RankTopology(rank,f2e,c2f,rel[0],rel[1],rel[2],maps,bc)


def adjacency_histogram(csr: CSRConnectivity) -> dict[int,int]:
    """Return adjacency valence counts, preserving unusual 3-cell edges."""
    vals=np.diff(csr.offsets); keys,counts=np.unique(vals,return_counts=True)
    return {int(k):int(c) for k,c in zip(keys,counts)}
