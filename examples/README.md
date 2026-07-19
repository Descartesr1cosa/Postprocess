# Examples

The examples are grouped by workflow rather than by individual API call:

- `case_diagnostics.py` contains case inspection and all static/dynamic checks.
- `reconstruct_B_cell.py` demonstrates magnetic-field reconstruction.
- `export_tecplot.py` writes the complete Tecplot output set.

For custom fields, use the public API rather than adding another exporter:

```python
from mpcns_post import MPCNSCase, export_fields_tecplot

case = MPCNSCase.load("/path/to/DATA_bin", data_dir="/path/to/DATA")
plane = case.select_plane(axis="y", value=0.0, tolerance=1e-8)
export_fields_tecplot(
    case,
    {
        "Na_fraction": case.fields["Na_plus_fraction"],
        "drift": case.fields["H_Na_drift_velocity"],
    },
    "/path/to/post_output/custom_y0.plt",
    selection=plane,
)
```

Run the scripts after installing the package with `pip install -e .`.

```bash
python examples/case_diagnostics.py /path/to/DATA_bin --data-dir /path/to/DATA

python examples/reconstruct_B_cell.py /path/to/DATA_bin --data-dir /path/to/DATA

python examples/export_tecplot.py /path/to/DATA_bin \
  --data-dir /path/to/DATA \
  --output-dir /path/to/post_output \
  --prefix mercury
```
