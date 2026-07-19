# MPCNS Mercury Python offline post-processing

`mpcns_post` reads, validates, and globally assembles MPCNS Mercury output in a
single Python process. It has no runtime or import dependency on the solver
source tree, MPI, or `~/MPCNS_Mercury`; the solver source was consulted only to
confirm the version-1 byte layout.

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
```

## Runnable Python examples

Inspection and validation are collected in one diagnostics script. Tecplot
export is also available as a standalone Python workflow, in addition to the
CLI command above.

```bash
python examples/case_diagnostics.py /path/to/DATA_bin \
  --data-dir /path/to/DATA

python examples/export_tecplot.py /path/to/DATA_bin \
  --data-dir /path/to/DATA \
  --output-dir /path/to/post_output \
  --prefix mercury
```

See [`examples/README.md`](examples/README.md) for the focused reconstruction
example and the complete example layout.

## Confirmed version-1 layouts

Static files have an 80-byte `MPCNSBIN` header. Every section has a 32-byte
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

```python
from mpcns_post import MPCNSCase

case = MPCNSCase.open("/path/to/DATA_bin")
restart = case.read_latest_restart(data_dir="/path/to/DATA")
fields = case.assemble_dynamic_fields(restart)
B_cell = case.reconstruct_B_cell(fields)
H = case.compute_primitive("H", fields)
print(case.summary(data_dir="/path/to/DATA"))
```

## Known limitations

Version 1 and little-endian float64/int64 data are supported. Each rank file is
the latest overwritten checkpoint, not a time-series archive. Regional
constant-B diagnostics for axis-touching and near-axis-shell cells are reported
as unavailable because the current v1 cell flags do not encode those regions.
No visualization, trajectory integration, magnetic field-line tracing, or MPI
Python execution is included in this stage.

The current bundled five-rank sample passes the complete validator. Its 54,150
Solid cells intentionally have inactive fluid species state, while all 216,600
Fluid cells are populated.

## Tecplot binary export

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
`--illuminated-frequency` and `--shadow-frequency`. Total B is induced B plus the saved additive field. Current is a
cell-centered curvilinear numerical curl of induced B, normalized by the
manifest `current_density_ref`; it is not the solver's unavailable mimetic
J-edge reconstruction. Each output is immediately read back to exact EOF, and
an `<prefix>_export_summary_<location>.json` is written beside the PLT files.
