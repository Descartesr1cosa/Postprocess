"""High-level, single-process MPCNS case API."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from .access import (
    Block,
    DerivedFieldRegistry,
    EntityView,
    FieldCollection,
    SpeciesData,
    base_location,
)
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
        self._constant_cache: dict[str, np.ndarray] = {}
        self._species_cache: dict[str, SpeciesData] = {}
        self.derived = DerivedFieldRegistry(self)
        self.fields = FieldCollection(self)
        self._register_builtin_fields()

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

    @classmethod
    def load(
        cls,
        case_directory: str | Path,
        *,
        data_dir: str | Path | None = None,
    ) -> "MPCNSCase":
        """Open static data and assemble the latest restart in one call."""
        case = cls.open(case_directory)
        case.load_latest(data_dir=data_dir)
        return case

    def load_latest(self, *, data_dir: str | Path | None = None) -> "MPCNSCase":
        """Load and assemble the latest restart, returning this case."""
        restarts = self.read_latest_restart(data_dir=data_dir)
        self.assemble_dynamic_fields(restarts)
        return self

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
        self._dynamic=assemble_dynamic(restarts,self.rank_topologies,self.geometry,self.manifest)
        self._species_cache.clear()
        self.derived.clear_cache()
        return self._dynamic

    @property
    def latest_restart(self):
        """The loaded rank restarts, or ``None`` before :meth:`load_latest`."""
        return self._latest

    @property
    def dynamic_fields(self):
        """Globally assembled restart fields, or ``None`` before loading."""
        return self._dynamic

    def _require_dynamic(self):
        if self._dynamic is None:
            raise ValidationError(
                "restart fields are not loaded; use MPCNSCase.load(...) or case.load_latest(...)"
            )
        return self._dynamic

    def entity(self, location: str) -> EntityView:
        """Return owner-only global geometry for cells, nodes, faces, or edges."""
        base = base_location(location)
        geometry = self.geometry
        if base == "cell":
            return EntityView(
                base,
                geometry.cell_gid,
                geometry.cell_center_xyz,
                geometry.cell_flags,
                geometry.cell_volume,
            )
        if base == "node":
            return EntityView(base, geometry.node_gid, geometry.node_xyz)
        if base == "face":
            return EntityView(
                base,
                geometry.face_gid,
                geometry.face_center_xyz,
                geometry.face_flags,
                geometry.face_area,
                geometry.face_area_vector,
            )
        return EntityView(
            base,
            geometry.edge_gid,
            geometry.edge_center_xyz,
            geometry.edge_flags,
            geometry.edge_length,
            connectivity=geometry.edge_node_ids,
        )

    @property
    def cells(self) -> EntityView:
        return self.entity("cell")

    @property
    def nodes(self) -> EntityView:
        return self.entity("node")

    @property
    def faces(self) -> EntityView:
        return self.entity("face")

    @property
    def edges(self) -> EntityView:
        return self.entity("edge")

    def connectivity(self, name: str):
        """Return a global-ID based CSR topology relation by name."""
        allowed = {
            "node_to_cell",
            "edge_to_cell",
            "face_to_cell",
            "cell_to_face",
        }
        if name not in allowed:
            raise KeyError(f"unknown connectivity {name!r}; choose from {sorted(allowed)}")
        relation = getattr(self.topology, name)
        if relation is None:
            raise ValidationError(f"connectivity {name!r} is unavailable")
        return relation

    @property
    def block_connections(self) -> dict[int, np.ndarray]:
        """Return raw inter-block connection tables keyed by rank."""
        return {
            topology.rank: topology.block_connections
            for topology in self.rank_topologies
        }

    def iter_blocks(self, location: str = "cell"):
        """Iterate structured maps with canonical indices and orientation data."""
        requested = location.lower()
        base = base_location(requested)
        global_index = GlobalIDIndex.build(self.entity(base).global_ids)
        cell_global_index = GlobalIDIndex.build(self.geometry.cell_gid)
        for topology in self.rank_topologies:
            cell_maps = {
                mapping.block_id: mapping
                for mapping in topology.local_maps
                if mapping.location == "cell"
            }
            for mapping in topology.local_maps:
                matches = (
                    mapping.location == requested
                    if requested not in {"face", "edge"}
                    else mapping.location.startswith(requested + "_")
                )
                if not matches:
                    continue
                indices = global_index.lookup(mapping.global_ids)
                physics = None
                cell_map = cell_maps.get(mapping.block_id)
                if cell_map is not None:
                    cell_indices = cell_global_index.lookup(cell_map.global_ids)
                    flags = np.unique(self.geometry.cell_flags[cell_indices])
                    if flags.size == 1:
                        flag = int(flags[0])
                        physics = next(
                            (
                                name.title()
                                for name, bit in self.manifest.cell_flag_bits.items()
                                if flag & int(bit)
                            ),
                            "Unknown",
                        )
                yield Block(
                    topology.rank,
                    mapping.block_id,
                    mapping.location,
                    mapping.logical_shape,
                    mapping.global_ids,
                    indices,
                    mapping.owner_mask,
                    mapping.orientation_sign,
                    physics,
                )

    def species(self, name: str) -> SpeciesData:
        """Return raw conserved, primitive, dimensional, and mask access for H/Na."""
        normalized = name.removesuffix("+")
        if normalized in self._species_cache:
            return self._species_cache[normalized]
        dynamic = self._require_dynamic()
        field_name = f"U_{normalized}"
        if field_name not in dynamic.fields:
            raise KeyError(f"species {name!r} is unavailable")
        primitive = self.compute_primitive(normalized, dynamic)
        data = SpeciesData(
            self,
            normalized,
            dynamic.fields[field_name],
            primitive.density,
            primitive.velocity,
            primitive.pressure,
            dynamic.valid_masks.get(
                field_name,
                np.ones(self.geometry.cell_gid.size, dtype=bool),
            ),
        )
        self._species_cache[normalized] = data
        return data

    @property
    def H(self) -> SpeciesData:
        return self.species("H")

    @property
    def Na(self) -> SpeciesData:
        return self.species("Na")

    def _register_builtin_fields(self) -> None:
        for species in ("H", "Na"):
            self.derived.register(
                f"{species}_density",
                lambda case, species=species: case.species(species).density,
                units="normalized",
            )
            self.derived.register(
                f"{species}_velocity",
                lambda case, species=species: case.species(species).velocity,
                units="normalized",
            )
            self.derived.register(
                f"{species}_pressure",
                lambda case, species=species: case.species(species).pressure,
                units="normalized",
            )
            self.derived.register(
                f"{species}_number_density_cm3",
                lambda case, species=species: case.species(species).number_density("cm^-3"),
                units="cm^-3",
            )
            self.derived.register(
                f"{species}_velocity_km_s",
                lambda case, species=species: case.species(species).velocity_in("km/s"),
                units="km/s",
            )
            self.derived.register(
                f"{species}_pressure_nPa",
                lambda case, species=species: case.species(species).pressure_in("nPa"),
                units="nPa",
            )
            self.derived.register(
                f"{species}_temperature_K",
                lambda case, species=species: case.species(species).temperature_kelvin,
                units="K",
            )

        def sodium_fraction(case):
            h = case.H.number_density()
            na = case.Na.number_density()
            total = h + na
            out = np.full(total.shape, np.nan)
            valid = case.H.valid_mask & case.Na.valid_mask & (total > 0)
            out[valid] = na[valid] / total[valid]
            return out

        self.derived.register(
            "Na_plus_fraction",
            sodium_fraction,
            units="1",
            description="n_Na+ / (n_H+ + n_Na+)",
        )
        self.derived.register(
            "H_Na_drift_velocity",
            lambda case: case.H.velocity - case.Na.velocity,
            units="normalized",
            description="u_H+ - u_Na+",
        )
        self.derived.register(
            "total_ion_pressure",
            lambda case: case.H.pressure + case.Na.pressure,
            units="normalized",
        )

    def register_derived_field(self, name, calculator=None, **metadata):
        """Register a per-case derived field directly or as a decorator."""
        overwrite = bool(metadata.get("overwrite", False))
        if not overwrite and name in self.available_fields:
            raise ValueError(f"field {name!r} is already available")
        return self.derived.register(name, calculator, **metadata)

    @property
    def available_fields(self) -> tuple[str, ...]:
        dynamic_names = (
            tuple(self._dynamic.fields)
            if self._dynamic is not None
            else tuple(
                str(field["name"])
                for field in self.manifest.existing_dynamic_data.get("fields", [])
            )
        )
        constant_names = tuple(str(field["name"]) for field in self.manifest.fields)
        return tuple(dict.fromkeys((*dynamic_names, *constant_names, *self.derived.definitions)))

    def field_location(self, name: str) -> str:
        if name in self.derived:
            return self.derived.location(name)
        specs = list(self.manifest.existing_dynamic_data.get("fields", [])) + list(
            self.manifest.fields
        )
        for spec in specs:
            if spec["name"] == name:
                return base_location(str(spec["location"]))
        raise KeyError(f"unknown field {name!r}")

    def _assemble_constant_field(self, name: str) -> np.ndarray:
        if name in self._constant_cache:
            return self._constant_cache[name]
        location = self.field_location(name)
        target = self.entity(location).global_ids
        index = GlobalIDIndex.build(target)
        chunks = [chunk for chunk in self.rank_constant_fields if name in chunk.fields]
        if not chunks:
            raise KeyError(f"constant field {name!r} is unavailable")
        values = np.concatenate([np.asarray(chunk.fields[name]) for chunk in chunks])
        global_ids = np.concatenate([chunk.global_ids[name] for chunk in chunks])
        if np.unique(global_ids).size != global_ids.size:
            raise ValidationError(f"duplicate constant-field owner IDs for {name}")
        shape = (target.size,) + values.shape[1:]
        assembled = np.full(shape, np.nan, dtype=np.float64)
        assembled[index.lookup(global_ids)] = values
        self._constant_cache[name] = assembled
        return assembled

    def get_field(self, name: str) -> np.ndarray:
        """Resolve one raw restart, static constant, or registered derived field."""
        if name in self.derived:
            return self.derived.evaluate(name)
        dynamic_names = {
            str(field["name"])
            for field in self.manifest.existing_dynamic_data.get("fields", [])
        }
        if name in dynamic_names:
            dynamic = self._require_dynamic()
            if name not in dynamic.fields:
                raise KeyError(f"field {name!r} is declared but absent from this restart")
            return dynamic.fields[name]
        if any(field["name"] == name for field in self.manifest.fields):
            return self._assemble_constant_field(name)
        raise KeyError(f"unknown field {name!r}")

    def field_global_ids(self, name: str) -> np.ndarray:
        """Return the exact global-ID ordering used by a public field."""
        if self._dynamic is not None and name in self._dynamic.global_ids:
            return self._dynamic.global_ids[name]
        return self.entity(self.field_location(name)).global_ids

    def valid_mask(self, name: str) -> np.ndarray:
        """Return the applicability/finite-row mask for any public field."""
        if self._dynamic is not None and name in self._dynamic.valid_masks:
            return self._dynamic.valid_masks[name]
        values = self.get_field(name)
        axes = tuple(range(1, values.ndim))
        return np.all(np.isfinite(values), axis=axes) if axes else np.isfinite(values)

    def select_plane(self, **kwargs):
        from .selection import select_plane

        return select_plane(self, **kwargs)

    def select_box(self, *args, **kwargs):
        from .selection import select_box

        return select_box(self, *args, **kwargs)

    def select_sphere(self, **kwargs):
        from .selection import select_sphere

        return select_sphere(self, **kwargs)

    def select_boundary_faces(self, **kwargs):
        from .surface import select_boundary_faces

        return select_boundary_faces(self, **kwargs)

    def species_flux(self, species: str, surface, **kwargs):
        from .surface import integrate_species_flux

        return integrate_species_flux(self, surface, species=species, **kwargs)

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

    def reconstruct_current_dec(self, *, validate_debug: bool = False):
        """Return Edge 1-form and Cell-vector current via the v3 DEC operators."""
        from .dec import reconstruct_current

        return reconstruct_current(self, validate_debug=validate_debug)

    def reconstruct_J_edge(self, *, physical: bool = False, validate_debug: bool = False):
        """Return DEC Edge ``J dot dr`` (normalized, or physical A/m)."""
        from .dec import reconstruct_current_edge

        return reconstruct_current_edge(
            self, physical=physical, validate_debug=validate_debug
        )

    def compute_current_dec(
        self,
        *,
        unit: str | None = "A/m^2",
        validate_debug: bool = False,
    ):
        """Return solver-equivalent Cartesian Cell current density."""
        from .dec import reconstruct_current_cell

        return reconstruct_current_cell(
            self, unit=unit, validate_debug=validate_debug
        )

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
            if self.reconstruction.B_face_to_J_edge is not None:
                has_debug = all(name in f.fields for name in ("J_xi", "J_eta", "J_zeta"))
                current = self.reconstruct_current_dec(validate_debug=has_debug)
                detail = (
                    f", debug J_edge max abs error={current.debug_edge_max_abs_error:.3e}"
                    if current.debug_edge_max_abs_error is not None else ""
                )
                report.add(
                    "info", "physics",
                    f"DEC J_cell |J|=[{np.linalg.norm(current.cell_vector,axis=1).min():.6g},"
                    f"{np.linalg.norm(current.cell_vector,axis=1).max():.6g}]{detail}",
                )
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
