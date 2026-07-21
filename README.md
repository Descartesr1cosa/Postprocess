# MPCNS Mercury Python offline post-processing

`mpcns_post` reads, validates, and globally assembles MPCNS Mercury output in a
single Python process. It has no runtime or import dependency on the solver
source tree, MPI, or `~/MPCNS_Mercury`. It supports legacy manifest v1 and the
expanded manifest v3 geometry, topology, and DEC reconstruction data.

## Install

```bash
cd python_post
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install pytest
```

NumPy is the only required dependency. Matplotlib and PyVista are optional and
are not used by the first-stage readers.

## Input layout

The static directory contains `manifest.json` and one `geometry_####.bin`,
`topology_####.bin`, `reconstruction_####.bin`, and
`constant_field_####.bin` per MPI rank. The dynamic directory contains the
existing `flow_field####.bin` checkpoint files. The manifest is the data entry
point: it identifies rank files, UUIDs, fields, units, operators, and oriented
face semantics.

```bash
python -m mpcns_post.cli inspect-manifest /path/to/DATA_bin
python -m mpcns_post.cli validate-static /path/to/DATA_bin
python -m mpcns_post.cli validate-case /path/to/DATA_bin --data-dir /path/to/DATA
python -m mpcns_post.cli diagnose /path/to/DATA_bin --data-dir /path/to/DATA
python -m mpcns_post.cli summary /path/to/DATA_bin --data-dir /path/to/DATA
python -m mpcns_post.cli export-tecplot /path/to/DATA_bin \
  --data-dir /path/to/DATA \
  --output-dir /path/to/post_output \
  --prefix mercury
python -m mpcns_post.cli export-fields /path/to/DATA_bin \
  Na_plus_fraction H_Na_drift_velocity \
  --data-dir /path/to/DATA \
  --output /path/to/post_output/custom.plt
```

## Runnable Python examples

The examples are deliberately limited to the unchanged supplied Node exporter
and one combined DATA/DEC validation script.

```bash
python examples/validate_case_dec.py /path/to/DATA_bin \
  --data-dir /path/to/DATA --validate-debug

# Edit its settings block first.
python examples/export_node_species_em.py
```

See [`examples/README.md`](examples/README.md) for the focused reconstruction
example and the complete example layout.

## Supported layouts

Static container files retain the version-1 80-byte `MPCNSBIN` header. Every section has a 32-byte
zero-padded name followed by scalar type, component count, entity count, byte
count, and a packed payload. The current topology writer emits a legacy naming
collision for the reverse-incidence `edge_global_id` and `face_global_id`
sections. The reader preserves the second occurrences internally as
`edge_cell_global_id` and `face_cell_global_id`; other duplicate names are
rejected.

Restart files are field-major: a field header is followed by all of that
field's blocks. C++ `FieldBlock` extents are half-open `[lo, hi)`. The writer
loops `i`, then `j`, then `k`, then component, so the normalized Python array is
directly `values[i,j,k,component]` with shape `(ni,nj,nk,ncomponents)`.

Topology structured maps cover interior physical entities and are linearized
with `i` fastest. Restart arrays include ghost layers; assembly indexes the
physical map coordinates through the stored restart extent and retains only
owner entities. It does not infer the physical region merely by trimming a
fixed ghost width.

Manifest v3 adds the complete restart B-face storage catalog, including raw
physical/interface/coupling ghosts, plus solver-materialized
`B_face_to_J_edge` and `J_edge_to_cell_cartesian` operators. Normal production
restarts contain only fluid state and induced B; the optional J-edge triplet is
accepted solely for validation.

The manifest may restrict a restart field to a physics domain. For example,
`U_H` and `U_Na` currently have `physics_domain="Fluid"`; inactive blocks whose
Cell flags are Solid are valid and remain `NaN` in full-size primitive arrays.
`GlobalFields.valid_masks` identifies applicable entities, and validation and
primitive statistics operate only on that mask. An inactive block inside its
declared domain remains a hard error.

Face and edge fields are oriented quantities. For dynamic face values the
manifest rule is applied exactly:

```text
global_owner_value = local_value / local_orientation_sign
```

Cell conserved variables are never orientation-transformed. The three face
families `B_xi`, `B_eta`, and `B_zeta` are assembled into one global face array,
then the stored `B_face_to_cell_cartesian` weights reconstruct Cartesian
cell-centered B. Current constant fields are `Badd_xi`, `Badd_eta`, `Badd_zeta`,
and `Photo_rate`; dynamic fields are `U_H`, `U_Na`, `B_xi`, `B_eta`, and
`B_zeta`.

## Python API

Open and assemble a complete case in one call. The older `open()`,
`read_latest_restart()`, and `assemble_dynamic_fields()` calls remain available
for applications that need explicit lifecycle control.

```python
from mpcns_post import MPCNSCase, export_fields_tecplot

case = MPCNSCase.load(
    "/path/to/DATA_bin",
    data_dir="/path/to/DATA",
)

# Owner-only, global-ID ordered entity data.
cell_xyz = case.cells.coordinates
cell_gid = case.cells.global_ids
face_area_vector = case.faces.area_vectors
face_to_cell = case.connectivity("face_to_cell")
block_connections = case.block_connections

# Raw conserved arrays, validity masks, primitives, and dimensional values.
u_na = case.fields["U_Na"]
fluid_mask = case.valid_mask("U_Na")
na = case.Na
na_number_cm3 = na.number_density("cm^-3")
na_velocity_km_s = na.velocity_in("km/s")
na_pressure_npa = na.pressure_in("nPa")
na_temperature = na.temperature_kelvin

# Block maps expose global indices, owner/orientation metadata, and reshaping.
for block in case.iter_blocks(location="cell"):
    block_pressure = block.reshape(case.fields["total_ion_pressure"])

# Every selection carries sparse indices plus full mask, coordinates, and IDs.
plane = case.select_plane(axis="y", value=0.0, tolerance=1.0e-8)
box = case.select_box(lower=(-2, -1, -1), upper=(2, 1, 1))
shell = case.select_sphere(radius=3.0, inner_radius=2.5)
```

### Registered derived fields

Derived fields are lazy, cached, and local to a case. Calculators receive the
case and return a global array whose leading dimension matches their declared
location. Built-ins include `Na_plus_fraction`, `H_Na_drift_velocity`, and
`total_ion_pressure`.

```python
import numpy as np

@case.register_derived_field(
    "ion_bulk_speed",
    location="cell",
    units="km/s",
)
def ion_bulk_speed(case):
    velocity = 0.5 * (case.H.velocity + case.Na.velocity)
    return np.linalg.norm(
        case.unit_converter.convert(velocity, "velocity", "km/s"),
        axis=1,
    )

speed = case.fields["ion_bulk_speed"]

export_fields_tecplot(
    case,
    {
        "Na_fraction": case.fields["Na_plus_fraction"],
        "ion_bulk_speed_km_s": speed,
        "H_Na_drift": case.fields["H_Na_drift_velocity"],
    },
    "/path/to/post_output/mercury_y0.plt",
    location="cell",
    selection=plane,
)
```

Axis-aligned plane/box subsets remain structured Tecplot sub-zones. Spherical
and oblique subsets remain separated by source block and are represented as
ordered point-strip zones. Vector fields are expanded to `_x`, `_y`, `_z`.

### Owner-only surface and escape flux

Boundary surfaces contain each global exterior Face exactly once. Outward area
vectors use the stored `cell_to_face` orientation sign and a geometric outward
check. The interpolation policy is explicit; exterior flux currently supports
only `owner`, so no ghost or MPI alias can be counted twice.

```python
# Select a geometrically defined part of the exterior boundary.
face_shell = case.select_sphere(
    center=(0, 0, 0),
    radius=4.0,
    inner_radius=3.9,
    location="face",
)
surface = case.select_boundary_faces(selection=face_shell)

# Positive total means net outward escape. Units are particles/s.
escape = case.species_flux(
    "Na",
    surface,
    kind="particle",
    dimensional=True,
    interpolation="owner",
)
print(escape.total, escape.units)
```

For a custom cell density/velocity pair, call `integrate_surface_flux`
directly. `kind="mass"` returns kg/s when `dimensional=True`.

The original explicit workflow remains valid:

```python
case = MPCNSCase.open("/path/to/DATA_bin")
restart = case.read_latest_restart(data_dir="/path/to/DATA")
fields = case.assemble_dynamic_fields(restart)

B_cell = case.reconstruct_B_cell(fields)
H = case.compute_primitive("H", fields)
```

### DEC current reconstruction

Manifest v3 cases reconstruct the same current path as Mercury from induced B
alone. Saved operators include final Hodge factors, orientations, alias
reduction, singular-edge treatment, boundary ghosts, and the pole override.
Prescribed `Badd_*` is excluded, matching the solver.

```python
case = MPCNSCase.load("/path/to/DATA_bin", data_dir="/path/to/DATA")

dec = case.reconstruct_current_dec()
J_edge_nd = dec.edge_1form       # normalized Edge J·dr
J_cell_nd = dec.cell_vector      # normalized Cartesian Cell J

J_cell_A_m2 = case.compute_current_dec(unit="A/m^2")
J_cell_nA_m2 = case.compute_current_dec(unit="nA/m^2")

# Only for a debug restart containing J_xi/J_eta/J_zeta:
checked = case.reconstruct_current_dec(validate_debug=True)
print(checked.debug_edge_max_abs_error)
```

## Known limitations

Manifest versions 1 and 3 and little-endian float64/int64 data are supported. Each rank file is
the latest overwritten checkpoint, not a time-series archive. Regional
constant-B diagnostics for axis-touching and near-axis-shell cells are reported
as unavailable because the current v1 cell flags do not encode those regions.
No interactive visualization, trajectory integration, magnetic field-line
tracing, or MPI Python execution is included in this stage.

The current bundled five-rank sample passes the complete validator. Its 54,150
Solid cells intentionally have inactive fluid species state, while all 216,600
Fluid cells are populated.

## Tecplot binary export

`export_fields_tecplot` is the general API for arbitrary cell/node/face/edge
arrays and spatial selections. The older `export_case_tecplot` and
`export-tecplot` CLI remain convenience workflows that construct the standard
Mercury fluid and electromagnetic output set described below.

`export-tecplot` is case-independent and writes Tecplot 112 binary ordered-zone
files without requiring Tecplot, PyTecplot, or MPI. Every rank-local structured
block becomes one named zone; all zones for the same physical group are merged
into one file. Zone names include rank, local block, and `Fluid`/`Solid`.

Node-centered structured output is the default. It applies the stored global
`cell_scalar_to_node` operator before splitting arrays into block zones, so the
same physical Node receives exactly the same value in every rank/zone alias.
At Fluid/Solid boundaries, Fluid quantities use only applicable Fluid Cells and
renormalize the remaining weights. The operator is CSR topology based: it does
not assume eight surrounding Cells, and singular-Edge endpoint Nodes use their
actual global incident-Cell rows. Retain cell-center output with
`--location cell`; use `--location node` explicitly if desired.

Two location-suffixed files are emitted:

- `<prefix>_fluid_node.plt` (or `_cell.plt`): neutral Na concentration, H+/Na+ number density,
  velocity, pressure, and temperature. Only Fluid zones are included.
- `<prefix>_electromagnetic_node.plt` (or `_cell.plt`): total magnetic field and current-density
  components for Fluid and Solid zones, plus `PhysicsCode`.

The units are encoded in variable names: coordinates in Mercury radii, number
density in cm^-3, velocity in km/s, pressure in nPa, temperature in K, magnetic
field in nT, and current density in nA/m^2. Neutral Na is recovered from the
saved photo-production field using the solver-defined illuminated/shadow
frequencies; different cases can override them with
`--illuminated-frequency` and `--shadow-frequency`. Total B is induced B plus the saved additive field. With v3 static
data, current uses the solver-equivalent DEC reconstruction from induced B;
legacy v1 data retains the cell-centered curl fallback because it has no DEC
operators. Each output is immediately read back to exact EOF, and
an `<prefix>_export_summary_<location>.json` is written beside the PLT files.
