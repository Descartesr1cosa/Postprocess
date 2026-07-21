# Examples

Only two focused scripts are retained:

- `export_node_species_em.py` retains the supplied export workflow; only its
  current helper now calls the DEC API.
- `validate_case_dec.py` is the only test script. It validates `DATA_bin` and
  `DATA`, reconstructs current from induced B, and can compare against optional
  debug J-edge fields.

```bash
python examples/validate_case_dec.py /path/to/DATA_bin \
  --data-dir /path/to/DATA

# For a debug restart containing J_xi/J_eta/J_zeta:
python examples/validate_case_dec.py /path/to/DATA_bin \
  --data-dir /path/to/DATA --validate-debug
```

Edit the settings section at the top of `export_node_species_em.py` before
running the full exporter.
