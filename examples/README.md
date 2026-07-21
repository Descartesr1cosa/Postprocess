# Examples

Only two focused scripts are retained:

- `dec_current.py` reconstructs current from induced B and can validate it
  against optional debug J-edge fields.
- `export_node_species_em.py` is the supplied complete Node species/EM export,
  upgraded so its current calculation calls the DEC API.

```bash
python examples/dec_current.py /path/to/DATA_bin \
  --data-dir /path/to/DATA

# For a debug restart containing J_xi/J_eta/J_zeta:
python examples/dec_current.py /path/to/DATA_bin \
  --data-dir /path/to/DATA --validate-debug
```

Edit the settings section at the top of `export_node_species_em.py` before
running the full exporter.
