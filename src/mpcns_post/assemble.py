"""Serial global-ID based assembly."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .errors import ValidationError
from .types import (CSRConnectivity, GlobalFields, GlobalGeometry, GlobalReconstruction,
    GlobalTopology, RankGeometry, RankReconstruction, RankRestart, RankTopology,
    ScalarReconstructionOperator, VectorReconstructionOperator)


@dataclass(frozen=True)
class GlobalIDIndex:
    gids: np.ndarray
    sorted_gids: np.ndarray
    order: np.ndarray

    @classmethod
    def build(cls,gids) -> "GlobalIDIndex":
        a=np.asarray(gids,dtype=np.int64); order=np.argsort(a)
        if a.size and np.any(np.diff(a[order])==0): raise ValidationError("duplicate global IDs")
        return cls(a,a[order],order.astype(np.int64))

    def lookup(self,query_gids) -> np.ndarray:
        """Map arbitrary global IDs to positions, rejecting missing IDs."""
        q=np.asarray(query_gids,dtype=np.int64); pos=np.searchsorted(self.sorted_gids,q)
        valid=pos<self.sorted_gids.size
        if np.any(valid): valid[valid] &= self.sorted_gids[pos[valid]]==q[valid]
        if not np.all(valid): raise ValidationError(f"unknown global ID {int(q[~valid][0])}")
        return self.order[pos]


def _entity(rank_geometries, prefix, attrs):
    ids=np.concatenate([getattr(g,prefix+"_gid") for g in rank_geometries])
    arrays=[np.concatenate([getattr(g,a) for g in rank_geometries],axis=0) for a in attrs]
    order=np.argsort(ids); ids=ids[order]; arrays=[a[order] for a in arrays]
    if ids.size and np.any(np.diff(ids)==0): raise ValidationError(f"duplicate owner {prefix} gid across ranks")
    return (ids,*arrays)


def assemble_geometry(chunks: list[RankGeometry]) -> GlobalGeometry:
    """Assemble owner-only geometry arrays in ascending global-ID order."""
    n=_entity(chunks,"node",["node_xyz"]); e=_entity(chunks,"edge",["edge_node_ids","edge_center_xyz","edge_dr","edge_length","edge_flags"])
    f=_entity(chunks,"face",["face_center_xyz","face_area_vector","face_area","face_flags"]); c=_entity(chunks,"cell",["cell_center_xyz","cell_volume","cell_flags"])
    return GlobalGeometry(*n,*e,*f,*c)


def _merge_reverse(topologies: list[RankTopology], attr: str) -> CSRConnectivity:
    rows={}
    for t in topologies:
        rel=getattr(t,attr); prefix=attr.split("_to_")[0]
        assert rel.row_global_ids is not None
        for i,gid in enumerate(rel.row_global_ids): rows.setdefault(int(gid),set()).update(map(int,rel.row(i)))
    off=[0]; vals=[]
    keys=np.asarray(sorted(rows),dtype=np.int64)
    for gid in keys: vals.extend(sorted(rows[int(gid)])); off.append(len(vals))
    return CSRConnectivity(np.asarray(off,dtype=np.int64),np.asarray(vals,dtype=np.int64),row_global_ids=keys)


def _merge_signed_rows(topologies: list[RankTopology], attr: str) -> CSRConnectivity:
    """Merge a global-ID keyed signed relation, rejecting sign conflicts."""
    rows: dict[int, dict[int, int]] = {}
    for topology in topologies:
        relation = getattr(topology, attr)
        assert relation.row_global_ids is not None
        assert relation.signs is not None
        for row_index, row_gid in enumerate(relation.row_global_ids):
            start, stop = relation.offsets[row_index:row_index + 2]
            row = rows.setdefault(int(row_gid), {})
            for entity_gid, sign in zip(
                relation.indices[start:stop],
                relation.signs[start:stop],
            ):
                previous = row.setdefault(int(entity_gid), int(sign))
                if previous != int(sign):
                    raise ValidationError(
                        f"conflicting {attr} orientation for row {int(row_gid)}, "
                        f"entity {int(entity_gid)}"
                    )

    row_gids = np.asarray(sorted(rows), dtype=np.int64)
    offsets = [0]
    indices: list[int] = []
    signs: list[int] = []
    for row_gid in row_gids:
        for entity_gid, sign in sorted(rows[int(row_gid)].items()):
            indices.append(entity_gid)
            signs.append(sign)
        offsets.append(len(indices))
    return CSRConnectivity(
        np.asarray(offsets, dtype=np.int64),
        np.asarray(indices, dtype=np.int64),
        np.asarray(signs, dtype=np.int32),
        row_global_ids=row_gids,
    )


def assemble_topology(chunks: list[RankTopology]) -> GlobalTopology:
    """Collect local maps and diagnostic reverse incidence relations."""
    return GlobalTopology(
        [m for t in chunks for m in t.local_maps],
        _merge_reverse(chunks,"node_to_cell"),
        _merge_reverse(chunks,"edge_to_cell"),
        _merge_reverse(chunks,"face_to_cell"),
        _merge_signed_rows(chunks,"cell_to_face"),
    )


def _merge_operator(chunks, attr, vector):
    grouped={}
    for r in chunks:
        op=getattr(r,attr)
        for i,gid in enumerate(op.output_global_ids):
            sl=slice(op.offsets[i],op.offsets[i+1]); grouped.setdefault(int(gid),[]).append((op.input_global_ids[sl],op.weights[sl]))
    outs=np.array(sorted(grouped),dtype=np.int64); offsets=[0]; inputs=[]; weights=[]
    for gid in outs:
        for ids,w in grouped[int(gid)]: inputs.append(ids); weights.append(w)
        offsets.append(offsets[-1]+sum(x[0].size for x in grouped[int(gid)]))
    inp=np.concatenate(inputs) if inputs else np.empty(0,dtype=np.int64)
    if weights: w=np.concatenate(weights,axis=0)
    else: w=np.empty((0,3),dtype=np.float64) if vector else np.empty(0,dtype=np.float64)
    cls=VectorReconstructionOperator if vector else ScalarReconstructionOperator
    return cls("B_face_to_cell_cartesian" if vector else "cell_scalar_to_node",outs,np.asarray(offsets,dtype=np.int64),inp,w)


def assemble_reconstruction(chunks: list[RankReconstruction]) -> GlobalReconstruction:
    """Merge owner-cell rows and distributed node rows by output global ID."""
    return GlobalReconstruction(_merge_operator(chunks,"B_face_to_cell",True),_merge_operator(chunks,"cell_scalar_to_node",False))


def assemble_dynamic(restarts: list[RankRestart], topologies: list[RankTopology], geometry: GlobalGeometry, manifest=None) -> GlobalFields:
    """Extract physical owner cells/faces from ghosted structured restart arrays."""
    if len(restarts)!=len(topologies): raise ValidationError("restart/topology rank count mismatch")
    # Current writer v1 geometry chunks contain quotient-owner Face rows only,
    # while structured topology maps and reconstruction use the full Face ID
    # space. Dynamic assembly therefore takes the authoritative full ID set
    # from the topology maps. This remains an ID-based operation (no coordinate
    # matching or topology inference).
    face_gids=np.unique(np.concatenate([m.global_ids for t in topologies for m in t.local_maps if m.location.startswith("face_")]))
    target={"U_H":geometry.cell_gid,"U_Na":geometry.cell_gid,"B_xi":face_gids,"B_eta":face_gids,"B_zeta":face_gids}
    out={}; indexes={k:GlobalIDIndex.build(v) for k,v in target.items()}; valid_masks={k:np.ones(v.size,dtype=bool) for k,v in target.items()}
    if manifest is not None:
        specs={str(x["name"]):x for x in manifest.existing_dynamic_data.get("fields",[])}
        fluid_bit=int(manifest.cell_flag_bits.get("fluid",0))
        for name in ("U_H","U_Na"):
            if specs.get(name,{}).get("physics_domain") == "Fluid" and fluid_bit:
                valid_masks[name]=(geometry.cell_flags & np.uint32(fluid_bit)) != 0
    for name,gids in target.items():
        comps=restarts[0].fields[name].components
        # Each B_* field owns only its matching stagger family; zeros on the
        # other two families make their sum the unified global Face field.
        fill=0.0 if name.startswith("B_") else np.nan
        out[name]=np.full((gids.size,comps),fill,dtype=np.float64)
    loc_by_field={"U_H":"cell","U_Na":"cell","B_xi":"face_xi","B_eta":"face_eta","B_zeta":"face_zeta"}
    inactive: dict[str,list[tuple[int,int]]]={name:[] for name in target}
    for rr,topo in zip(restarts,topologies):
        maps={(m.block_id,m.location):m for m in topo.local_maps}
        for name,loc in loc_by_field.items():
            field=rr.fields[name]
            for bi,block in enumerate(field.blocks):
                if not block.extent.active:
                    inactive[name].append((rr.rank,bi)); continue
                m=maps.get((bi,loc))
                if m is None: raise ValidationError(f"rank {rr.rank}: missing block {bi} map for {loc}")
                ni,nj,nk=m.logical_shape; lo=block.extent.lo
                sl=tuple(slice(-lo[a],-lo[a]+m.logical_shape[a]) for a in range(3))
                vals=block.values[sl].reshape((-1,field.components),order="F")
                # Topology writer linearizes k/j/i (i fastest), while restart
                # arrays are represented in native i/j/k axes.
                gids=m.global_ids; mask=m.owner_mask
                idx=indexes[name].lookup(gids[mask]); selected=vals[mask]
                if loc.startswith("face_"): selected=selected/m.orientation_sign[mask,None]
                out[name][idx]=selected
    for name,a in out.items():
        applicable=valid_masks[name]
        if np.any(~np.all(np.isfinite(a[applicable]),axis=1)):
            bad_rows=np.flatnonzero(applicable & ~np.all(np.isfinite(a),axis=1))
            raise ValidationError(f"global field {name} has {bad_rows.size} unfilled/non-finite entities (first indices {bad_rows[:5].tolist()}); inactive rank/block entries={inactive[name][:12]}")
        if a.shape[1]==1: out[name]=a[:,0]
    return GlobalFields(out,target,valid_masks)
