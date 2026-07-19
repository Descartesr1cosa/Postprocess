# Examples

The examples are grouped by workflow rather than by individual API call:

- `case_diagnostics.py` contains case inspection and all static/dynamic checks.
- `reconstruct_B_cell.py` demonstrates magnetic-field reconstruction.
- `export_tecplot.py` writes the complete Tecplot output set.

Run the scripts after installing the package with `pip install -e .`.

```bash
python examples/case_diagnostics.py /path/to/DATA_bin --data-dir /path/to/DATA

python examples/reconstruct_B_cell.py /path/to/DATA_bin --data-dir /path/to/DATA

python examples/export_tecplot.py /path/to/DATA_bin \
  --data-dir /path/to/DATA \
  --output-dir /path/to/post_output \
  --prefix mercury
```
