"""ASCII Tecplot time history and readable summary writer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def write_outputs(rows: list[dict], output_dir: Path) -> tuple[Path, Path]:
    """Write any registered scalar Q values, in addition to time and Nstep."""
    if not rows:
        raise ValueError("No time samples were processed")
    variables = tuple(rows[0])
    if variables[:2] != ("time", "Nstep"):
        raise ValueError("Each output row must begin with time and Nstep")
    if any(tuple(row) != variables for row in rows):
        raise ValueError("Every output row must contain the same Q variables")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dat_path = output_dir / "convergence_time_history.dat"
    with dat_path.open("w", encoding="ascii", newline="\n") as stream:
        stream.write('TITLE = "MPCNS subsolar convergence time history"\n')
        stream.write("VARIABLES = " + ", ".join(f'\"{name}\"' for name in variables) + "\n")
        stream.write(f"ZONE T=\"time history\", I={len(rows)}, F=POINT\n")
        for row in rows:
            stream.write(" ".join(f"{float(row[name]):.12e}" for name in variables) + "\n")

    summary = {"sample_count": len(rows), "quantities": {}}
    for name in variables[2:]:
        values = np.asarray([row[name] for row in rows], dtype=float)
        finite = values[np.isfinite(values)]
        summary["quantities"][name] = {
            "unit": "Mercury radii (R_M)",
            "valid_sample_count": int(finite.size),
            "mean": float(np.mean(finite)) if finite.size else None,
            "standard_deviation": float(np.std(finite)) if finite.size else None,
        }
    json_path = output_dir / "convergence_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return dat_path, json_path
