"""High-level, single-process MPCNS case API."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from .assemble import GlobalIDIndex,assemble_dynamic,assemble_geometry,assemble_reconstruction,assemble_topology
from .constant_field import read_rank_constant_fields
from .derived import UnitConverter,conserved_to_primitive
from .errors import ValidationError
from .geometry import read_rank_geometry
from .manifest import load_manifest
from .reconstruction import read_rank_reconstruction
from .restart import read_rank_restart
from .topology import adjacency_histogram,read_rank_topology
from .validation import ValidationReport,finite_statistics,validate_constant_B_reproduction

class MPCNSCase:
    """An eagerly loaded static case with lazy latest-restart data."""
    def __init__(self,directory,manifest,geometries,topologies,reconstructions,constant_fields):
        self.directory=Path(directory); self.manifest=manifest
        self.rank_geometries=geometries; self.rank_topologies=topologies; self.rank_reconstructions=reconstructions; self.rank_constant_fields=constant_fields
        self.geometry=assemble_geometry(geometries); self.topology=assemble_topology(topologies); self.reconstruction=assemble_reconstruction(reconstructions)
        self.unit_converter=UnitConverter(manifest.normalization); self._latest=None; self._dynamic=None

    @classmethod
    def open(cls,case_directory: str|Path) -> "MPCNSCase":
        """Open manifest-listed static chunks without solver dependencies."""
        d=Path(case_directory); m=load_manifest(d)
        gs=[]; ts=[]; rs=[]; cs=[]
        for rank in range(m.number_of_ranks):
            gs.append(read_rank_geometry(d/m.files["geometry"][rank],rank=rank,manifest=m))
            ts.append(read_rank_topology(d/m.files["topology"][rank],rank=rank,manifest=m))
            rs.append(read_rank_reconstruction(d/m.files["reconstruction"][rank],rank=rank,manifest=m))
            cs.append(read_rank_constant_fields(d/m.files["constant_field"][rank],rank=rank,manifest=m))
        return cls(d,m,gs,ts,rs,cs)

    def _data_dir(self,data_dir):
        if data_dir is not None: return Path(data_dir)
        sibling=self.directory.parent/"DATA"
        return sibling if sibling.is_dir() else self.directory

    def read_latest_restart(self, *, data_dir: str|Path|None=None):
        """Read every rank's latest checkpoint and verify step/time/schema agreement."""
        d=self._data_dir(data_dir); pattern=str(self.manifest.existing_dynamic_data["path_pattern"])
        name=Path(pattern.replace("{rank:04d}","0000")).name
        prefix=name.replace("0000","")
        restarts=[read_rank_restart(d/(name.replace("0000",f"{r:04d}")),rank=r,manifest=self.manifest) for r in range(self.manifest.number_of_ranks)]
        first=restarts[0]
        for x in restarts[1:]:
            if x.step!=first.step or not np.isclose(x.time,first.time,rtol=0,atol=1e-13) or x.version!=first.version or tuple(x.fields)!=tuple(first.fields): raise ValidationError("rank restart step/time/version/field mismatch")
        self._latest=restarts; return restarts

    read_restart_step=read_latest_restart

    def assemble_dynamic_fields(self,restarts=None):
        """Assemble owner physical cells/faces into global arrays."""
        if restarts is None: restarts=self._latest
        if restarts is None: raise ValidationError("read a restart before dynamic assembly")
        self._dynamic=assemble_dynamic(restarts,self.rank_topologies,self.geometry,self.manifest); return self._dynamic

    def reconstruct_B_cell(self,fields=None):
        """Add dynamic face fields and reconstruct global Cartesian cell B."""
        fields=fields or self._dynamic
        if fields is None: raise ValidationError("assemble dynamic fields first")
        face_gids=fields.global_ids.get("B_xi",self.geometry.face_gid); face=np.zeros(face_gids.size)
        for n in ("B_xi","B_eta","B_zeta"): face+=fields.fields[n]
        op=self.reconstruction.B_face_to_cell; local=op.apply(face,GlobalIDIndex.build(face_gids))
        out=np.full((self.geometry.cell_gid.size,3),np.nan); out[GlobalIDIndex.build(self.geometry.cell_gid).lookup(op.output_global_ids)]=local
        if not np.all(np.isfinite(out)): raise ValidationError("B reconstruction left unfilled cells")
        return out

    def compute_primitive(self,species: str,fields=None):
        """Compute H or Na primitive variables from global conserved state."""
        fields=fields or self._dynamic
        if fields is None: raise ValidationError("assemble dynamic fields first")
        name="U_"+species; q=fields.fields[name]; mask=fields.valid_masks.get(name,np.ones(q.shape[0],dtype=bool))
        valid=conserved_to_primitive(q[mask],gamma=self.manifest.physical_constants["gamma"])
        density=np.full(q.shape[0],np.nan); velocity=np.full((q.shape[0],3),np.nan); pressure=np.full(q.shape[0],np.nan)
        density[mask]=valid.density; velocity[mask]=valid.velocity; pressure[mask]=valid.pressure
        return type(valid)(density,velocity,pressure,None)

    def validate_static(self) -> ValidationReport:
        """Run cross-file topology and reconstruction checks."""
        report=ValidationReport()
        missing_by_kind={}
        location_sets={"node":{"node"},"edge":{"edge_xi","edge_eta","edge_zeta"},"face":{"face_xi","face_eta","face_zeta"},"cell":{"cell"}}
        for kind,locations in location_sets.items():
            full=np.unique(np.concatenate([m.global_ids for t in self.rank_topologies for m in t.local_maps if m.location in locations]))
            stored=getattr(self.geometry,kind+"_gid"); missing=np.setdiff1d(full,stored); missing_by_kind[kind]=missing
            if missing.size: report.add("error","global_id",f"geometry lacks {missing.size} of {full.size} topology {kind} IDs (first={int(missing[0])})")
        try:
            face_index=GlobalIDIndex.build(self.geometry.face_gid); face_index.lookup(self.reconstruction.B_face_to_cell.input_global_ids)
        except Exception as exc: report.add("error","reconstruction",str(exc))
        sums=np.add.reduceat(self.reconstruction.cell_scalar_to_node.weights,self.reconstruction.cell_scalar_to_node.offsets[:-1]) if self.reconstruction.cell_scalar_to_node.output_global_ids.size else np.array([])
        max_sum=float(np.max(np.abs(sums-1))) if sums.size else 0.
        hist=adjacency_histogram(self.topology.edge_to_cell)
        report.add("info","topology",f"edge adjacency histogram {hist}; 3-cell edges={hist.get(3,0)}")
        report.add("info","reconstruction",f"node scalar max |weight sum-1|={max_sum:.3e}")
        if not missing_by_kind["face"].size:
            try:
                rb=validate_constant_B_reproduction(self.geometry,self.reconstruction)
                severity="error" if rb.max_absolute_error>1e-10 else "info"
                report.add(severity,"reconstruction",f"constant-B max absolute error={rb.max_absolute_error:.3e}")
            except Exception as exc: report.add("error","reconstruction",str(exc))
        else:
            report.add("warning","reconstruction","constant-B reproduction unavailable because directed Face geometry is incomplete")
        return report

    def validate_all(self, *, data_dir: str|Path|None=None) -> ValidationReport:
        """Validate static data, latest restart, assembly, primitives, and B."""
        report=self.validate_static()
        try:
            rr=self.read_latest_restart(data_dir=data_dir); f=self.assemble_dynamic_fields(rr); b=self.reconstruct_B_cell(f)
            for s in ("H","Na"):
                p=self.compute_primitive(s,f); valid=np.isfinite(p.density)
                report.add("info","physics",f"{s}: applicable cells={np.count_nonzero(valid)}, inactive cells={np.count_nonzero(~valid)}, rho=[{np.nanmin(p.density):.6g},{np.nanmax(p.density):.6g}], pressure=[{np.nanmin(p.pressure):.6g},{np.nanmax(p.pressure):.6g}], nonpositive pressure={np.count_nonzero(p.pressure[valid]<=0)}")
            report.add("info","physics",f"B_cell |B|=[{np.linalg.norm(b,axis=1).min():.6g},{np.linalg.norm(b,axis=1).max():.6g}]")
        except Exception as exc: report.add("error","restart",str(exc))
        return report

    def summary(
        self,
        *,
        data_dir: str | Path | None = None,
        include_restart: bool = True,
    ) -> str:
        """Return a concise human-readable static or complete case summary."""
        g=self.geometry; lines=[f"case UUID: {self.manifest.case_uuid}",f"mesh UUID: {self.manifest.mesh_uuid}",f"ranks: {self.manifest.number_of_ranks}",f"blocks: {self.manifest.number_of_blocks}",f"global entities: nodes={g.node_gid.size}, edges={g.edge_gid.size}, faces={g.face_gid.size}, cells={g.cell_gid.size}",f"edge adjacency: {adjacency_histogram(self.topology.edge_to_cell)}"]
        if include_restart:
            try:
                rr=self._latest or self.read_latest_restart(data_dir=data_dir); f=self._dynamic or self.assemble_dynamic_fields(rr)
                lines.append(f"latest step/time: {rr[0].step} / {rr[0].time:.17g}")
                for name,a in f.fields.items(): st=finite_statistics(a); lines.append(f"{name}: min={st['min']:.6g}, max={st['max']:.6g}")
            except Exception as exc: lines.append(f"latest restart: unavailable ({exc})")
        try:
            rb=validate_constant_B_reproduction(g,self.reconstruction); lines.append(f"constant-B max error: {rb.max_absolute_error:.3e}")
        except Exception as exc: lines.append(f"constant-B validation: unavailable ({exc})")
        return "\n".join(lines)
