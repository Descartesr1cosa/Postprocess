"""Tecplot 112 binary output for globally assembled structured MPCNS blocks."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import struct
from typing import Mapping, Sequence

import numpy as np

from .assemble import GlobalIDIndex
from .access import Selection, base_location
from .derived import curvilinear_curl, neutral_sodium_from_photo_rate, species_temperature_kelvin
from .errors import BinaryFormatError, ValidationError


@dataclass(frozen=True)
class TecplotZone:
    """One structured ordered zone; every variable has shape (ni,nj,nk)."""
    name: str
    physics: str
    values: Mapping[str,np.ndarray]


@dataclass(frozen=True)
class TecplotFileInfo:
    path: Path
    title: str
    variables: tuple[str,...]
    zones: tuple[tuple[str,tuple[int,int,int]],...]


def _write_i32(fp,value: int) -> None: fp.write(struct.pack("<i",int(value)))
def _write_f32(fp,value: float) -> None: fp.write(struct.pack("<f",float(value)))
def _write_f64(fp,value: float) -> None: fp.write(struct.pack("<d",float(value)))
def _write_string(fp,value: str) -> None:
    try: encoded=value.encode("ascii")
    except UnicodeEncodeError as exc: raise ValueError(f"Tecplot names must be ASCII: {value!r}") from exc
    for byte in encoded: _write_i32(fp,byte)
    _write_i32(fp,0)


def write_tecplot_binary(path: str|Path, *, title: str, variable_names: Sequence[str], zones: Sequence[TecplotZone], solution_time: float=0.0) -> Path:
    """Write a Tecplot 112 FULL ordered-zone binary file atomically."""
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True); names=tuple(variable_names)
    if not names or len(names)!=len(set(names)) or not zones: raise ValueError("Tecplot output requires unique variables and at least one zone")
    checked=[]
    for zone in zones:
        if set(zone.values)!=set(names): raise ValidationError(f"zone {zone.name}: variable set mismatch")
        arrays={n:np.asarray(zone.values[n],dtype=np.float64) for n in names}; shapes={a.shape for a in arrays.values()}
        if len(shapes)!=1: raise ValidationError(f"zone {zone.name}: variable shapes differ")
        shape=next(iter(shapes))
        if len(shape)!=3 or min(shape)<=0 or any(np.any(~np.isfinite(a)) for a in arrays.values()): raise ValidationError(f"zone {zone.name}: invalid shape or non-finite data")
        checked.append((zone,arrays,shape))
    tmp=p.with_suffix(p.suffix+".tmp")
    try:
        with tmp.open("wb") as fp:
            fp.write(b"#!TDV112"); _write_i32(fp,1); _write_i32(fp,0); _write_string(fp,title); _write_i32(fp,len(names))
            for name in names: _write_string(fp,name)
            for zone,arrays,shape in checked:
                _write_f32(fp,299.0); _write_string(fp,zone.name); _write_i32(fp,-1); _write_i32(fp,-2); _write_f64(fp,solution_time); _write_i32(fp,-1); _write_i32(fp,0)
                _write_i32(fp,1)
                for _ in names: _write_i32(fp,0)  # all arrays are nodal to the exported structured grid
                _write_i32(fp,0); _write_i32(fp,0)
                for size in shape: _write_i32(fp,size)
                _write_i32(fp,0)  # no zone AuxData; physics is encoded in zone name/PhysicsCode
            _write_f32(fp,357.0)
            for zone,arrays,shape in checked:
                _write_f32(fp,299.0)
                for _ in names: _write_i32(fp,1)  # float32 payload
                _write_i32(fp,0); _write_i32(fp,0); _write_i32(fp,-1)
                for name in names: _write_f64(fp,float(arrays[name].min())); _write_f64(fp,float(arrays[name].max()))
                for name in names: fp.write(np.asarray(arrays[name].ravel(order="F"),dtype="<f4").tobytes())
        tmp.replace(p)
    except Exception:
        if tmp.exists(): tmp.unlink()
        raise
    return p


class _TecReader:
    def __init__(self,path): self.path=Path(path); self.data=self.path.read_bytes(); self.pos=0
    def read(self,n):
        end=self.pos+n
        if end>len(self.data): raise BinaryFormatError(f"{self.path} at {self.pos}: truncated Tecplot file")
        out=self.data[self.pos:end]; self.pos=end; return out
    def i32(self): return struct.unpack("<i",self.read(4))[0]
    def f32(self): return struct.unpack("<f",self.read(4))[0]
    def f64(self): return struct.unpack("<d",self.read(8))[0]
    def string(self):
        chars=[]
        while True:
            value=self.i32()
            if value==0: break
            if not 0<value<128: raise BinaryFormatError(f"{self.path} at {self.pos-4}: invalid Tecplot string")
            chars.append(value)
        return bytes(chars).decode("ascii")


def inspect_tecplot_binary(path: str|Path) -> TecplotFileInfo:
    """Read back the subset emitted by :func:`write_tecplot_binary` to EOF."""
    r=_TecReader(path)
    if r.read(8)!=b"#!TDV112" or r.i32()!=1 or r.i32()!=0: raise BinaryFormatError(f"{r.path}: unsupported Tecplot header")
    title=r.string(); nvar=r.i32(); variables=tuple(r.string() for _ in range(nvar)); zones=[]
    while True:
        marker=r.f32()
        if marker==357.0: break
        if marker!=299.0: raise BinaryFormatError(f"{r.path}: invalid zone marker {marker}")
        name=r.string(); r.i32(); r.i32(); r.f64(); r.i32()
        if r.i32()!=0 or r.i32()!=1: raise BinaryFormatError(f"{r.path}: expected ordered zone with variable locations")
        for _ in range(nvar): r.i32()
        r.i32(); r.i32(); shape=(r.i32(),r.i32(),r.i32())
        while r.i32()!=0: r.string(); r.i32(); r.string()
        zones.append((name,shape))
    for _,shape in zones:
        if r.f32()!=299.0: raise BinaryFormatError(f"{r.path}: invalid data marker")
        types=[r.i32() for _ in range(nvar)]
        if any(x!=1 for x in types) or r.i32()!=0 or r.i32()!=0 or r.i32()!=-1: raise BinaryFormatError(f"{r.path}: unsupported data section")
        for _ in range(nvar):
            lo,hi=r.f64(),r.f64()
            if not np.isfinite(lo+hi) or lo>hi: raise BinaryFormatError(f"{r.path}: invalid variable range")
        r.read(4*nvar*int(np.prod(shape,dtype=np.int64)))
    if r.pos!=len(r.data): raise BinaryFormatError(f"{r.path}: {len(r.data)-r.pos} trailing bytes")
    return TecplotFileInfo(r.path,title,variables,tuple(zones))


def _scalar_tecplot_fields(
    fields: Mapping[str, np.ndarray],
    entity_count: int,
) -> dict[str, np.ndarray]:
    """Validate global arrays and expand vector/tensor components to scalars."""
    scalar_fields: dict[str, np.ndarray] = {}
    component_labels = ("x", "y", "z")
    for name, values in fields.items():
        array = np.asarray(values)
        if array.ndim == 0 or array.shape[0] != entity_count:
            raise ValidationError(
                f"field {name!r} must have leading size {entity_count}, got {array.shape}"
            )
        if array.ndim == 1:
            scalar_fields[str(name)] = array
            continue
        flattened = array.reshape((entity_count, -1))
        for component in range(flattened.shape[1]):
            suffix = (
                component_labels[component]
                if flattened.shape[1] == 3
                else str(component)
            )
            scalar_fields[f"{name}_{suffix}"] = flattened[:, component]
    if not scalar_fields:
        raise ValueError("at least one field is required")
    return scalar_fields


def _selected_zone_values(
    block,
    global_values: Mapping[str, np.ndarray],
    selection_mask: np.ndarray | None,
) -> dict[str, np.ndarray] | None:
    arrays = {name: block.reshape(values) for name, values in global_values.items()}
    if selection_mask is None:
        return arrays
    local_mask = block.reshape(selection_mask)
    positions = np.argwhere(local_mask)
    if positions.size == 0:
        return None

    lower = positions.min(axis=0)
    upper = positions.max(axis=0) + 1
    slices = tuple(slice(int(lo), int(hi)) for lo, hi in zip(lower, upper))
    rectangular = bool(np.all(local_mask[slices])) and int(np.prod(upper - lower)) == positions.shape[0]
    if rectangular:
        return {name: values[slices] for name, values in arrays.items()}

    # A sphere or oblique plane is not a tensor-product subset. It remains one
    # zone per source block, represented as an ordered point strip.
    return {
        name: values[local_mask].reshape((-1, 1, 1))
        for name, values in arrays.items()
    }


def export_fields_tecplot(
    case,
    fields: Mapping[str, np.ndarray],
    output_path: str | Path,
    *,
    location: str = "cell",
    selection: Selection | None = None,
    title: str = "MPCNS fields",
) -> TecplotFileInfo:
    """Export arbitrary global fields while retaining one zone per block.

    Vector fields are expanded to ``_x/_y/_z`` variables. Axis-aligned plane
    and box selections remain structured sub-zones. Non-tensor-product regions
    (for example a sphere) remain separated by block and are written as ordered
    point strips with shape ``(N, 1, 1)``.
    """
    normalized_location = base_location(location)
    entities = case.entity(normalized_location)
    if selection is not None:
        if selection.location != normalized_location or selection.mask.shape != (
            entities.size,
        ):
            raise ValueError("selection location/order does not match export location")

    scalar_fields = _scalar_tecplot_fields(fields, entities.size)
    all_values: dict[str, np.ndarray] = {
        "X": entities.coordinates[:, 0],
        "Y": entities.coordinates[:, 1],
        "Z": entities.coordinates[:, 2],
        **scalar_fields,
    }
    if len(all_values) != len(scalar_fields) + 3:
        raise ValueError("field names X, Y, and Z are reserved for coordinates")

    zones = []
    for block in case.iter_blocks(normalized_location):
        zone_values = _selected_zone_values(
            block,
            all_values,
            None if selection is None else selection.mask,
        )
        if zone_values is None:
            continue
        zone_name = (
            f"rank{block.rank:04d}_block{block.block_id:04d}_"
            f"{block.location}_{block.physics or 'Unknown'}"
        )
        zones.append(TecplotZone(zone_name, block.physics or "Unknown", zone_values))
    if not zones:
        raise ValidationError("Tecplot selection contains no structured block entities")

    solution_time = 0.0
    if case.latest_restart:
        solution_time = float(case.latest_restart[0].time)
    path = write_tecplot_binary(
        output_path,
        title=title,
        variable_names=tuple(all_values),
        zones=zones,
        solution_time=solution_time,
    )
    return inspect_tecplot_binary(path)


def _assemble_constant(case,name: str) -> tuple[np.ndarray,np.ndarray]:
    gids=np.concatenate([chunk.global_ids[name] for chunk in case.rank_constant_fields]); vals=np.concatenate([chunk.fields[name] for chunk in case.rank_constant_fields])
    order=np.argsort(gids); gids=gids[order]; vals=vals[order]
    if gids.size and np.any(np.diff(gids)==0): raise ValidationError(f"duplicate constant-field owner IDs for {name}")
    return gids,vals


def project_cell_values_to_nodes(case, values: np.ndarray, *, valid_mask: np.ndarray|None=None) -> np.ndarray:
    """Project global Cell values to global Nodes with shared-ID continuity.

    The stored scalar reconstruction is applied component-wise. If a physics
    mask is supplied, links to inapplicable Cells are omitted and the remaining
    weights are renormalized at each Node.
    """
    source=np.asarray(values,dtype=np.float64); scalar=source.ndim==1
    if scalar: source=source[:,None]
    if source.ndim!=2 or source.shape[0]!=case.geometry.cell_gid.size: raise ValidationError("Cell-to-Node source shape mismatch")
    valid=np.ones(source.shape[0],dtype=bool) if valid_mask is None else np.asarray(valid_mask,dtype=bool)
    if valid.shape!=(source.shape[0],): raise ValidationError("Cell-to-Node validity mask shape mismatch")
    op=case.reconstruction.cell_scalar_to_node; cell_links=GlobalIDIndex.build(case.geometry.cell_gid).lookup(op.input_global_ids)
    rows=np.repeat(np.arange(op.output_global_ids.size,dtype=np.int64),np.diff(op.offsets)); link_valid=valid[cell_links]
    local=np.full((op.output_global_ids.size,source.shape[1]),np.nan,dtype=np.float64)
    for component in range(source.shape[1]):
        finite=link_valid&np.isfinite(source[cell_links,component]); effective=op.weights*finite
        component_denom=np.bincount(rows,weights=effective,minlength=op.output_global_ids.size)
        numerator=np.bincount(rows,weights=effective*np.where(finite,source[cell_links,component],0.0),minlength=op.output_global_ids.size)
        component_good=component_denom>0; local[component_good,component]=numerator[component_good]/component_denom[component_good]
    out=np.full((case.geometry.node_gid.size,source.shape[1]),np.nan,dtype=np.float64)
    out[GlobalIDIndex.build(case.geometry.node_gid).lookup(op.output_global_ids)]=local
    return out[:,0] if scalar else out


def _block_records(case):
    node_blocks = {
        (block.rank, block.block_id): block
        for block in case.iter_blocks(location="node")
    }
    seen = 0
    for cell_block in case.iter_blocks(location="cell"):
        key = (cell_block.rank, cell_block.block_id)
        try:
            node_block = node_blocks[key]
        except KeyError as exc:
            raise ValidationError(
                f"rank {cell_block.rank} block {cell_block.block_id}: missing Node map"
            ) from exc
        flags = case.geometry.cell_flags[cell_block.indices]
        unique = np.unique(flags)
        if unique.size != 1:
            raise ValidationError(
                f"rank {cell_block.rank} block {cell_block.block_id}: mixed physics "
                "flags are not supported in one ordered zone"
            )
        yield (
            cell_block.rank,
            cell_block.block_id,
            cell_block.physics or "Unknown",
            cell_block.logical_shape,
            cell_block.indices,
            node_block.logical_shape,
            node_block.indices,
        )
        seen += 1
    if seen!=case.manifest.number_of_blocks: raise ValidationError(f"local block count {seen} != manifest {case.manifest.number_of_blocks}")


def export_case_tecplot(case, *, data_dir: str|Path|None=None, output_dir: str|Path, prefix: str="mpcns", location: str="node", illuminated_frequency: float=5.0e-5, shadow_frequency: float=1.0e-5) -> dict[str,object]:
    """Export two globally merged structured Tecplot files at Nodes or Cells."""
    if location not in {"node","cell"}: raise ValueError("location must be 'node' or 'cell'")
    output=Path(output_dir); output.mkdir(parents=True,exist_ok=True)
    restarts=case.read_latest_restart(data_dir=data_dir); fields=case.assemble_dynamic_fields(restarts)
    h=case.compute_primitive("H",fields); na=case.compute_primitive("Na",fields)
    b_ind=case.reconstruct_B_cell(fields)
    face_index=GlobalIDIndex.build(fields.global_ids["B_xi"]); badd_face=np.zeros(face_index.gids.size)
    for name in ("Badd_xi","Badd_eta","Badd_zeta"):
        gids,vals=_assemble_constant(case,name); badd_face[face_index.lookup(gids)]=vals
    op=case.reconstruction.B_face_to_cell; badd_local=op.apply(badd_face,face_index); badd=np.full_like(b_ind,np.nan)
    badd[GlobalIDIndex.build(case.geometry.cell_gid).lookup(op.output_global_ids)]=badd_local
    if np.any(~np.isfinite(badd)): raise ValidationError("additive magnetic reconstruction left unfilled cells")
    b_total=b_ind+badd
    photo_gids,photo_values=_assemble_constant(case,"Photo_rate"); photo=np.full(case.geometry.cell_gid.size,np.nan); photo[GlobalIDIndex.build(case.geometry.cell_gid).lookup(photo_gids)]=photo_values
    fluid_bit=int(case.manifest.cell_flag_bits["fluid"]); fluid=(case.geometry.cell_flags&np.uint32(fluid_bit))!=0
    if np.any(~np.isfinite(photo[fluid])): raise ValidationError("Photo_rate is incomplete on Fluid cells")
    neutral=np.full(photo.shape,np.nan); neutral[fluid]=neutral_sodium_from_photo_rate(photo[fluid],case.geometry.cell_center_xyz[fluid],illuminated_frequency=illuminated_frequency,shadow_frequency=shadow_frequency)
    units=case.unit_converter; n_h=units.number_density(h.density,particle_mass=case.manifest.physical_constants["particle_mass_H"],unit="cm^-3")
    n_na=units.number_density(na.density,particle_mass=case.manifest.physical_constants["particle_mass_Na"],unit="cm^-3")
    t_h=species_temperature_kelvin(h.density,h.pressure,density_ref=case.manifest.normalization["density_ref"],pressure_ref=case.manifest.normalization["pressure_ref"],particle_mass=case.manifest.physical_constants["particle_mass_H"])
    t_na=species_temperature_kelvin(na.density,na.pressure,density_ref=case.manifest.normalization["density_ref"],pressure_ref=case.manifest.normalization["pressure_ref"],particle_mass=case.manifest.physical_constants["particle_mass_Na"])
    fluid_global={"X_RM":case.geometry.cell_center_xyz[:,0],"Y_RM":case.geometry.cell_center_xyz[:,1],"Z_RM":case.geometry.cell_center_xyz[:,2],"Na_neutral_cm-3":neutral,"Hplus_n_cm-3":n_h,"Hplus_Ux_km-s":units.convert(h.velocity[:,0],"velocity","km/s"),"Hplus_Uy_km-s":units.convert(h.velocity[:,1],"velocity","km/s"),"Hplus_Uz_km-s":units.convert(h.velocity[:,2],"velocity","km/s"),"Hplus_p_nPa":units.convert(h.pressure,"pressure","nPa"),"Hplus_T_K":t_h,"Naplus_n_cm-3":n_na,"Naplus_Ux_km-s":units.convert(na.velocity[:,0],"velocity","km/s"),"Naplus_Uy_km-s":units.convert(na.velocity[:,1],"velocity","km/s"),"Naplus_Uz_km-s":units.convert(na.velocity[:,2],"velocity","km/s"),"Naplus_p_nPa":units.convert(na.pressure,"pressure","nPa"),"Naplus_T_K":t_na}
    em_global={"X_RM":case.geometry.cell_center_xyz[:,0],"Y_RM":case.geometry.cell_center_xyz[:,1],"Z_RM":case.geometry.cell_center_xyz[:,2],"B_total_x_nT":units.convert(b_total[:,0],"magnetic_field","nT"),"B_total_y_nT":units.convert(b_total[:,1],"magnetic_field","nT"),"B_total_z_nT":units.convert(b_total[:,2],"magnetic_field","nT")}
    blocks=list(_block_records(case)); current_global=np.full_like(b_ind,np.nan)
    if case.reconstruction.B_face_to_J_edge is not None:
        current_global = case.compute_current_dec(unit="nA/m^2")
        current_method = "solver-equivalent DEC: Jedge=M1^-1 D1^T M2 Bind, then Jedge-to-cell"
    else:
        # Retain v1 export compatibility for static data that predates the DEC
        # operators.  New v3 output always takes the branch above.
        for rank,block,physics,cell_shape,cell_idx,node_shape,node_idx in blocks:
            reshape_cell=lambda a:np.asarray(a[cell_idx]).reshape(cell_shape,order="F")
            coords=np.stack([reshape_cell(case.geometry.cell_center_xyz[:,c]) for c in range(3)],axis=-1); bind=np.stack([reshape_cell(b_ind[:,c]) for c in range(3)],axis=-1)
            curl=curvilinear_curl(coords,bind); current=units.convert(curl,"current_density","nA/m^2")
            current_global[cell_idx]=current.reshape((-1,3),order="F")
        current_method = "legacy v1 fallback: cell-centered curvilinear curl of induced B"
    current_range = [float(np.min(current_global)), float(np.max(current_global))]
    em_global.update({"J_induced_x_nA-m2":current_global[:,0],"J_induced_y_nA-m2":current_global[:,1],"J_induced_z_nA-m2":current_global[:,2],"PhysicsCode":case.geometry.cell_flags.astype(np.float64)})
    if np.any(~np.isfinite(current_global)): raise ValidationError("current-density assembly left unfilled Cells")
    if location=="node":
        fluid_output={"X_RM":case.geometry.node_xyz[:,0],"Y_RM":case.geometry.node_xyz[:,1],"Z_RM":case.geometry.node_xyz[:,2]}
        for name,value in fluid_global.items():
            if name not in fluid_output: fluid_output[name]=project_cell_values_to_nodes(case,value,valid_mask=fluid)
        em_output={"X_RM":case.geometry.node_xyz[:,0],"Y_RM":case.geometry.node_xyz[:,1],"Z_RM":case.geometry.node_xyz[:,2]}
        for name,value in em_global.items():
            if name not in em_output and name!="PhysicsCode": em_output[name]=project_cell_values_to_nodes(case,value)
    else:
        fluid_output=fluid_global; em_output=em_global
    fluid_zones=[]; em_zones=[]
    for rank,block,physics,cell_shape,cell_idx,node_shape,node_idx in blocks:
        zone_name=f"rank{rank:04d}_block{block:04d}_{physics}"
        if location=="node": shape=node_shape; idx=node_idx
        else: shape=cell_shape; idx=cell_idx
        reshape=lambda a:np.asarray(a[idx]).reshape(shape,order="F")
        em_values={name:reshape(value) for name,value in em_output.items()}
        em_values["PhysicsCode"]=np.full(shape,case.geometry.cell_flags[cell_idx][0],dtype=np.float64)
        em_zones.append(TecplotZone(zone_name,physics,em_values))
        if physics=="Fluid": fluid_zones.append(TecplotZone(zone_name,physics,{name:reshape(value) for name,value in fluid_output.items()}))
    fluid_path=write_tecplot_binary(output/f"{prefix}_fluid_{location}.plt",title=f"MPCNS fluid species ({location})",variable_names=tuple(fluid_output),zones=fluid_zones,solution_time=restarts[0].time)
    em_names=tuple(em_zones[0].values); em_path=write_tecplot_binary(output/f"{prefix}_electromagnetic_{location}.plt",title=f"MPCNS electromagnetic fields ({location})",variable_names=em_names,zones=em_zones,solution_time=restarts[0].time)
    fluid_info=inspect_tecplot_binary(fluid_path); em_info=inspect_tecplot_binary(em_path)
    def zone_ranges(zones,names):
        return {name:[float(min(np.min(z.values[name]) for z in zones)),float(max(np.max(z.values[name]) for z in zones))] for name in names}
    summary={"case_uuid":case.manifest.case_uuid,"mesh_uuid":case.manifest.mesh_uuid,"step":restarts[0].step,"time":restarts[0].time,"location":location,"files":{"fluid":{"path":str(fluid_path),"bytes":fluid_path.stat().st_size,"zones":len(fluid_info.zones),"variables":list(fluid_info.variables),"ranges":zone_ranges(fluid_zones,fluid_info.variables)},"electromagnetic":{"path":str(em_path),"bytes":em_path.stat().st_size,"zones":len(em_info.zones),"variables":list(em_info.variables),"ranges":zone_ranges(em_zones,em_info.variables)}},"neutral_sodium_method":f"Photo_rate / nu; illuminated nu={illuminated_frequency:.17g} s^-1, shadow nu={shadow_frequency:.17g} s^-1","current_method":current_method,"current_nA_m2_range":current_range}
    (output/f"{prefix}_export_summary_{location}.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    return summary
