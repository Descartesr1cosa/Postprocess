"""Variable-width Tecplot history writer for slice X/O-point coordinates."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def topology_coordinate_summary(rows: list[dict]) -> dict:
    """Summarise the representative tail/dayside topology coordinates."""
    groups = {
        "tail_x": [],
        "dayside_x_start": [],
        "dayside_x_end": [],
        "dayside_o_middle": [],
    }
    for row in rows:
        points = row["points"]
        tail_x = [point for point in points if point.kind == "X" and point.xyz_RM[0] < 0.0]
        dayside = [point for point in points if point.xyz_RM[0] >= 0.0]
        dayside_x = [point for point in dayside if point.kind == "X"]
        dayside_o = [point for point in dayside if point.kind == "O"]
        # If multiple tail sites exist, use the one nearest Mercury, which is
        # normally the principal near-tail reconnection X-point.
        if tail_x:
            groups["tail_x"].append(max(tail_x, key=lambda point: point.xyz_RM[0]).xyz_RM)
        if dayside_x:
            groups["dayside_x_start"].append(dayside_x[0].xyz_RM)
        if len(dayside_x) >= 2:
            groups["dayside_x_end"].append(dayside_x[-1].xyz_RM)
        if dayside_o:
            # The O point nearest the middle of the already ordered dayside
            # X/O chain is the requested central O-point.
            index_by_id = {id(point): index for index, point in enumerate(dayside)}
            middle = 0.5 * (len(dayside) - 1)
            centre_o = min(dayside_o, key=lambda point: abs(index_by_id[id(point)] - middle))
            groups["dayside_o_middle"].append(centre_o.xyz_RM)

    labels = {
        "tail_x": "tail X-point (x<0), nearest Mercury when several exist",
        "dayside_x_start": "first X-point in the ordered dayside topology chain",
        "dayside_x_end": "last X-point in the ordered dayside topology chain",
        "dayside_o_middle": "O-point nearest the middle of the ordered dayside topology chain",
    }
    result = {"unit": "Mercury radii (R_M)", "definitions": labels, "coordinates": {}}
    for name, values in groups.items():
        array = np.asarray(values, dtype=float)
        result["coordinates"][name] = {
            "valid_sample_count": int(len(array)),
            "mean_xyz_RM": None if not len(array) else np.mean(array, axis=0).tolist(),
            "variance_xyz_RM2": None if not len(array) else np.var(array, axis=0).tolist(),
            "standard_deviation_xyz_RM": None if not len(array) else np.std(array, axis=0).tolist(),
        }
    return result


def write_plane_topology(rows: list[dict], output_dir: Path) -> tuple[Path, Path]:
    """Write zero-padded point coordinates, with columns sized for max count."""
    if not rows:
        raise ValueError("No slice-topology samples were processed")
    max_points = max(len(row["points"]) for row in rows)
    variables = ["time", "Nstep"] + [
        name for index in range(max_points)
        for name in (f"x_{index}", f"y_{index}", f"z_{index}", f"type_{index}")
    ]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dat_path = output_dir / "plane_xo_points.dat"
    with dat_path.open("w", encoding="ascii", newline="\n") as stream:
        stream.write('TITLE = "MPCNS Cartesian-plane X/O-point cell centres"\n')
        stream.write("# type_i: -1 = not found; 0 = X-point; 1 = O-point\n")
        stream.write("# point slots: tail X-points first, then dayside X/O topology chain\n")
        stream.write("VARIABLES = " + ", ".join(f'\"{name}\"' for name in variables) + "\n")
        stream.write(f"ZONE T=\"plane X/O history\", I={len(rows)}, F=POINT\n")
        for row in rows:
            values = [float(row["time"]), float(row["Nstep"])]
            for point in row["points"]:
                values.extend(np.asarray(point.xyz_RM, dtype=float))
                values.append(0.0 if point.kind == "X" else 1.0)
            for _ in range(max_points - len(row["points"])):
                values.extend((0.0, 0.0, 0.0, -1.0))
            stream.write(" ".join(f"{value:.12e}" for value in values) + "\n")

    detail = {
        "maximum_point_count": max_points,
        "type_encoding": {"-1": "not found", "0": "X-point", "1": "O-point"},
        "slot_order": "tail X-points first, then dayside X/O topology chain; uses a monotonic in-plane coordinate when possible, otherwise an X/O-aware nearest-neighbour chain",
        "samples": [
            {"time": row["time"], "Nstep": row["Nstep"],
             "points": [{"type": point.kind, "xyz_RM": point.xyz_RM.tolist()} for point in row["points"]]}
            for row in rows
        ],
    }
    json_path = output_dir / "plane_xo_points_detail.json"
    json_path.write_text(json.dumps(detail, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return dat_path, json_path
