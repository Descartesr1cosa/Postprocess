# Bundled five-rank validation result

Validated on 2026-07-14 against `../out/DATA_bin` and `../out/DATA`.

```text
case UUID: 2e646544c2fb8da18573919e1af0dbcc
mesh UUID: 4e43bb79b26d56acdb1164667dae910c
ranks / blocks: 5 / 12
nodes: 273168
edges: 816832
faces: 814416
cells: 270750
latest step / time: 1000 / 0.041119814759071942
```

Result: **PASS**. Every static and restart file decodes through exact EOF;
headers, UUIDs, payload sizes, section metadata, rank step/time, field metadata,
global IDs, geometry, topology, reconstruction, and dynamic physical values all
pass validation.

Numerical checks:

```text
edge adjacent-cell histogram: {2: 8664, 3: 1000, 4: 807168}
3-cell edges accepted: 1000
node scalar max |sum(weights)-1|: 1.110e-16
constant Cartesian B maximum reproduction error: 7.866e-12

H applicable/inactive cells: 216600 / 54150
H rho range:      [0.203168, 2.35484]
H pressure range: [0.00850667, 0.441858]
H nonpositive pressure count: 0

Na applicable/inactive cells: 216600 / 54150
Na rho range:      [9.9969e-07, 0.24731]
Na pressure range: [2.58164e-11, 0.000582881]
Na nonpositive pressure count: 0

B_cell |B| range: [7.77227e-05, 59.4836]
```

The 54,150 inactive species cells are all Solid (`cell_flags=2`). This matches
the manifest's `physics_domain="Fluid"` and `inactive_semantics`; they are not
missing fluid data. All 216,600 Fluid cells (`cell_flags=1`) contain valid H and
Na states.

Reproduce with:

```bash
cd python_post
python -m mpcns_post.cli validate-case ../out/DATA_bin --data-dir ../out/DATA
python -m mpcns_post.cli summary ../out/DATA_bin --data-dir ../out/DATA
```
