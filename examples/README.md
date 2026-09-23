# Examples

## Split post-processing example

The `post/` directory is a complete, modular MPCNS post-processing example.
Use `post/run_post.py` as the only normal entry point.

Before running it, edit the settings at the top of these files:

- `post/run_post.py`: set `CASE_DIR` to the directory that contains `DATA` and
  `DATA_bin`, then choose output modules with `RUN_NODE_TECPLT`,
  `RUN_NA_ALTITUDE_PROFILES`, and `RUN_VIRTUAL_FLIGHT`.
- `post/post_virtual_flight.py`: set `MESSENGER_DATA_DIR` to the directory
  containing the 2008 MESSENGER PDS3 `.LBL` and `.TAB` files when virtual
  flight is enabled.
- `post/messenger_mso_data.py`: set `DATA_DIR` only when running this helper
  by itself to select/average MAG data and write a CSV file.

Run from the project root:

```bash
python examples/post/run_post.py
```

The modules have separate responsibilities:

- `post_case_io.py`: read `DATA_bin` and `DATA`.
- `post_core.py`: shared H+/Na+, electromagnetic, DEC-current, Cell, and Node
  calculations.
- `post_tecplot.py`: standard Fluid Node Tecplot output.
- `post_na_altitude.py`: Na+ altitude profiles and their reference models.
- `post_virtual_flight.py`: MESSENGER trajectory sampling and virtual-flight
  Tecplot output.
- `messenger_mso_data.py`: standalone MESSENGER MAG MSO reader/averager.

## Grid/time convergence example

`converge/` processes either one current output (`DATA_bin` + `DATA`) or every
archived output (`DATA_bin` + `DATA_archive/Step_*_Time_*`).  It retains static
data only, reads one `flow_field####.bin` time directory at a time, and writes
an ASCII Tecplot time history plus JSON mean/standard-deviation summary to
`DATA_DIR/tecplot_output`.

Edit `converge/run_converge.py` and run:

```bash
python examples/converge/run_converge.py
```

The initial quantities are the subsolar magnetopause x position (5-cell
locally averaged DEC-current maximum) and bow-shock x position (outermost
fast-magnetosonic Mach-one crossing).  The subsolar tube radius, x range, and
3--5-cell local average are explicit settings at the top of the runner.

The same runner can also write `plane_xo_points.dat` for one `x/y/z=value`
cell-centred slab.  It reports tail X-points first and then dayside X/O points
as a topology chain: an in-plane monotonic coordinate is used when possible,
otherwise nearest neighbours with alternating X/O type are preferred.  Each
`x_i,y_i,z_i` group has `type_i` (`0=X`, `1=O`) and is zero-padded to the
largest point count found in the full time sequence.  Configure the plane near
`RUN_PLANE_TOPOLOGY` in `converge/run_converge.py`.

Its X/O classification uses true structured-grid four-Cell contours of the
in-plane magnetic components (for `y=y0`, `Bz,Bx` versus `z,x`), a local 2-D
Jacobian, its trace/determinant discriminant, and phase winding.  This avoids
treating every small-|B| Cell as a topological point.
Adjacent quadrilateral detections are merged by their net winding within the
configured `PLANE_MERGE_RADIUS_IN_SPACINGS`.

`PLANE_POINT_POSITION_MODE` selects the coordinate output: `cell_center`
reports the existing representative Cell centre, while `interpolated` reports
the local affine `B_t=0` estimate in the accepted four-Cell contour.

`converge/converge_na_statistics.py` adds full-fluid-domain, volume-weighted
Na+ density mean/standard deviation and signed outward Na+ particle fluxes
through configurable virtual spheres.  Set `NA_FLUX_RADII_RM` and
`NA_SPHERE_SAMPLE_COUNT` in `run_converge.py`; the quantities are added to the
same convergence Tecplot history and JSON summary.

## Resumable time processing

Every completed time step is immediately checkpointed as a JSON file under
`DATA_DIR/temp_post_data`, named with its `Nstep` and `Time`.  On later runs,
compatible cached steps are verified and reused without opening their dynamic
`flow_field####.bin` files.  The final files in `tecplot_output` are built from
the union of `DATA_archive` times and valid `temp_post_data` cache times.

To validate a case without the full example workflow:

```bash
python examples/validate_case_dec.py /path/to/DATA_bin \
  --data-dir /path/to/DATA

# For a debug restart containing J_xi/J_eta/J_zeta:
python examples/validate_case_dec.py /path/to/DATA_bin \
  --data-dir /path/to/DATA --validate-debug
```

## Time-series analysis example

`analysis/` is the incremental science-analysis workflow.  It follows the
same memory-safe lifecycle as `converge/`: static `DATA_bin` is loaded once;
each archived flow field is loaded, analysed, checkpointed, released, and then
the next time is opened.  Edit `analysis/run_analysis.py` and run:

```bash
python examples/analysis/run_analysis.py
```

Its `OPERATIONS` list is a one-line switchboard: comment an operation line to
disable it.  The first implementation writes these scalar quantities:

- subsolar `R_MP`, `R_BS`, and resolved bow-shock jumps in `|B|`, total ion
  pressure, and H+ density (downstream minus upstream);
- total Na+ inventory and total photoionization source rate;
- surface loss/return and outer-boundary escape/inflow, with the signed net
  retained for each;
- net, outward-only, and inward-only Na+ virtual-sphere fluxes at every radius
  in `NA_SPHERE_RADII_RM`;
- a configurable circular tail-plane flux, with tailward and sunward parts
  kept distinct;
- configurable geometric near-surface, cusp, mantle, plasma-sheet, tail, and
  dayside magnetosheath-proxy region inventories plus topology-defined
  enclosing-boundary net/outward/inward fluxes and per-region y/z asymmetry;
- full-domain `y>0/y<0` and `z>0/z<0` Na+ inventories and asymmetry
  `A=(N_+-N_-)/(N_++N_-)`.

Products are isolated under `DATA_DIR/tecplot_output/analysis/`:

- `scalars/analysis_time_history.dat` and `analysis_summary.json` are the
  reusable time/Nstep scalar results;
- `cache/Nstep_*_Time_*.json` is the resumable per-time record;
- `volume/<time>.plt` is a complete binary Tecplot file per output time.  It
  contains all Fluid blocks at globally shared **Nodes**: Cell fields are
  projected with the same global Cell-to-Node reconstruction as `post/`, so
  interfaces connect continuously.  Variables include `X/Y/Z`, bulk `U`,
  H+/Na+/total density and pressure, total `B`, Cartesian `J`, `|J|`,
  `J_perp`, FAC, and the three requested Na+ composition ratios.  Create any
  desired slice interactively in Tecplot from this 3-D data.

The runner normally processes all available times.  For an inexpensive single
time smoke test, run it with `MPCNS_ANALYSIS_MAX_TIME_SAMPLES=1` in the
environment.

`sodium_load_factor` is immediately below `DATA_DIR`: `0.0` disables the Na
photoionization source, `1.0` retains the saved baseline, and larger values
apply sensitivity scaling.  The saved source is
`Q_src=sodium_load_factor*integral(Photo_rate dV)`.

The final scalar history additionally contains `Na_inventory_time_derivative_particles_s`
and `Na_budget_residual_particles_s = dot(N) - Q_src + Phi_surface_net +
Phi_outer_net`.  The signed net surface/outer-boundary fluxes are required for
closure; positive-only `loss` and `escape` remain separate one-way reporting
diagnostics.
Configure `QUASI_STEADY_WINDOWS` in `run_analysis.py`; each inclusive window
gets mean, standard deviation, minimum, and maximum in both
`analysis_summary.json` and `analysis_quasi_steady_windows.dat`.  Add existing
case directories to `CROSS_CASE_DATA_DIRS` to write a combined long-form
`cross_case_window_summary.csv`.
