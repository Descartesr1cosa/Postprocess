"""API-first serial post-processing for MPCNS Mercury."""

from .access import (
    Block,
    DerivedField,
    DerivedFieldRegistry,
    EntityView,
    FieldCollection,
    Selection,
    SpeciesData,
)
from .case import MPCNSCase
from .derived import PrimitiveSpecies, UnitConverter, conserved_to_primitive
from .manifest import load_manifest
from .restart import read_rank_restart
from .selection import select_box, select_plane, select_sphere
from .surface import (
    FluxResult,
    SurfaceSelection,
    integrate_species_flux,
    integrate_surface_flux,
    select_boundary_faces,
)
from .tecplot import (
    export_case_tecplot,
    export_fields_tecplot,
    inspect_tecplot_binary,
)

__all__ = [
    "Block",
    "DerivedField",
    "DerivedFieldRegistry",
    "EntityView",
    "FieldCollection",
    "FluxResult",
    "MPCNSCase",
    "PrimitiveSpecies",
    "Selection",
    "SpeciesData",
    "SurfaceSelection",
    "UnitConverter",
    "conserved_to_primitive",
    "export_case_tecplot",
    "export_fields_tecplot",
    "inspect_tecplot_binary",
    "integrate_species_flux",
    "integrate_surface_flux",
    "load_manifest",
    "read_rank_restart",
    "select_boundary_faces",
    "select_box",
    "select_plane",
    "select_sphere",
]

__version__ = "0.2.0"
