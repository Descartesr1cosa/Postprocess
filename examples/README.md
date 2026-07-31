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

To validate a case without the full example workflow:

```bash
python examples/validate_case_dec.py /path/to/DATA_bin \
  --data-dir /path/to/DATA

# For a debug restart containing J_xi/J_eta/J_zeta:
python examples/validate_case_dec.py /path/to/DATA_bin \
  --data-dir /path/to/DATA --validate-debug
```
