"""Writers for scalar/vector time-history analysis products."""

from __future__ import annotations

import json
import csv
from pathlib import Path

import numpy as np


def _statistics(values: np.ndarray) -> dict[str, float | int | None]:
    finite = values[np.isfinite(values)]
    return {
        "valid_sample_count": int(finite.size),
        "mean": float(np.mean(finite)) if finite.size else None,
        "standard_deviation": float(np.std(finite)) if finite.size else None,
        "minimum": float(np.min(finite)) if finite.size else None,
        "maximum": float(np.max(finite)) if finite.size else None,
    }


def _window_summary(rows: list[dict[str, float]], variables: tuple[str, ...], units: dict[str, str],
                    windows: tuple[tuple[float, float], ...]) -> tuple[dict, list[dict[str, float]]]:
    """Return JSON-friendly and Tecplot-row summaries for configured windows."""
    summary, table_rows = {}, []
    for index, (start, stop) in enumerate(windows):
        if not np.isfinite(start) or not np.isfinite(stop) or start > stop:
            raise ValueError("Each quasi-steady window must have finite t_min <= t_max")
        selected = [row for row in rows if start <= float(row["time"]) <= stop]
        item = {"t_min": float(start), "t_max": float(stop), "sample_count": len(selected), "quantities": {}}
        flat = {"window_index": float(index), "t_min": float(start), "t_max": float(stop), "sample_count": float(len(selected))}
        for name in variables[2:]:
            stats = _statistics(np.asarray([row[name] for row in selected], dtype=float)) if selected else _statistics(np.array([], dtype=float))
            item["quantities"][name] = {"unit": units.get(name, "unspecified"), **stats}
            for output_name, suffix in (("mean", "mean"), ("standard_deviation", "std"), ("minimum", "min"), ("maximum", "max")):
                flat[f"{name}_{suffix}"] = float("nan") if stats[output_name] is None else float(stats[output_name])
        summary[f"window_{index:03d}"] = item
        table_rows.append(flat)
    return summary, table_rows


def write_scalar_outputs(rows: list[dict[str, float]], output_dir: Path, *, units: dict[str, str],
                         quasi_steady_windows: tuple[tuple[float, float], ...] = ()) -> tuple[Path, Path, Path | None]:
    """Write one Tecplot ASCII time history and a compact JSON summary."""
    if not rows:
        raise ValueError("No time samples were processed")
    variables = tuple(rows[0])
    if variables[:2] != ("time", "Nstep"):
        raise ValueError("Rows must begin with time and Nstep")
    if any(tuple(row) != variables for row in rows):
        raise ValueError("Every time sample must have identical quantities")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dat_path = output_dir / "analysis_time_history.dat"
    with dat_path.open("w", encoding="ascii", newline="\n") as stream:
        stream.write('TITLE = "MPCNS time-series analysis"\n')
        stream.write("VARIABLES = " + ", ".join(f'\"{name}\"' for name in variables) + "\n")
        stream.write(f"ZONE T=\"time history\", I={len(rows)}, F=POINT\n")
        for row in rows:
            stream.write(" ".join(f"{float(row[name]):.12e}" for name in variables) + "\n")
    summary = {"sample_count": len(rows), "quantities": {}}
    for name in variables[2:]:
        values = np.asarray([row[name] for row in rows], dtype=float)
        summary["quantities"][name] = {"unit": units.get(name, "unspecified"), **_statistics(values)}
    windows, window_rows = _window_summary(rows, variables, units, quasi_steady_windows)
    summary["quasi_steady_windows"] = windows
    json_path = output_dir / "analysis_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    window_path = None
    if window_rows:
        window_path = output_dir / "analysis_quasi_steady_windows.dat"
        names = tuple(window_rows[0])
        with window_path.open("w", encoding="ascii", newline="\n") as stream:
            stream.write('TITLE = "MPCNS quasi-steady window statistics"\n')
            stream.write("VARIABLES = " + ", ".join(f'\"{name}\"' for name in names) + "\n")
            stream.write(f"ZONE T=\"configured windows\", I={len(window_rows)}, F=POINT\n")
            for row in window_rows:
                stream.write(" ".join(f"{row[name]:.12e}" for name in names) + "\n")
    return dat_path, json_path, window_path


def write_cross_case_window_summary(case_data_dirs: tuple[Path, ...], output_path: Path) -> Path:
    """Collect configured-window statistics from multiple already-processed cases."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for case_dir in map(Path, case_data_dirs):
        summary_path = case_dir / "tecplot_output" / "analysis" / "scalars" / "analysis_summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(f"Cross-case summary needs existing analysis output: {summary_path}")
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        for window_name, window in payload.get("quasi_steady_windows", {}).items():
            for quantity, stats in window["quantities"].items():
                rows.append({
                    "case": case_dir.name,
                    "window": window_name,
                    "t_min": window["t_min"], "t_max": window["t_max"],
                    "quantity": quantity, "unit": stats["unit"],
                    "sample_count": stats["valid_sample_count"],
                    "mean": stats["mean"], "standard_deviation": stats["standard_deviation"],
                    "minimum": stats["minimum"], "maximum": stats["maximum"],
                })
    fieldnames = ("case", "window", "t_min", "t_max", "quantity", "unit", "sample_count", "mean", "standard_deviation", "minimum", "maximum")
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path
