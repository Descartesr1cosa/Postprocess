"""Per-time-step JSON cache for resumable convergence post-processing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from converge_plane_topology import PlanePoint


CACHE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CachedTime:
    path: Path
    step: int
    time: float
    row: dict
    plane_row: dict | None


def _cache_filename(step: int, time: float) -> str:
    return f"Nstep_{step:09d}_Time_{time:.12e}.json"


def _json_value(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {name: _json_value(item) for name, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_value(item) for item in value]
    return value


def _float_value(value) -> float:
    return float("nan") if value is None else float(value)


def write_time_cache(cache_dir: Path, row: dict, plane_row: dict | None, *, configuration: dict) -> Path:
    """Immediately checkpoint all processed scalar and plane-topology data."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    step, time = int(row["Nstep"]), float(row["time"])
    plane = None
    if plane_row is not None:
        plane = {
            "coordinate_mode": plane_row["coordinate_mode"],
            "points": [
                {"type": point.kind, "xyz_RM": _json_value(point.xyz_RM)}
                for point in plane_row["points"]
            ],
        }
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "configuration": configuration,
        "Nstep": step,
        "time": time,
        "row": _json_value(row),
        "plane_topology": plane,
    }
    path = cache_dir / _cache_filename(step, time)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def read_time_cache(path: Path, *, configuration: dict) -> CachedTime | None:
    """Read a compatible cache; return None for stale/incomplete cache files."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload["schema_version"] != CACHE_SCHEMA_VERSION or payload["configuration"] != configuration:
            return None
        row = {name: _float_value(value) for name, value in payload["row"].items()}
        step, time = int(payload["Nstep"]), float(payload["time"])
        if int(row["Nstep"]) != step or not np.isclose(row["time"], time, rtol=0.0, atol=1.0e-12):
            return None
        plane = payload.get("plane_topology")
        plane_row = None
        if plane is not None:
            plane_row = {
                "time": time,
                "Nstep": step,
                "coordinate_mode": str(plane["coordinate_mode"]),
                "points": [
                    PlanePoint(str(point["type"]), np.asarray(point["xyz_RM"], dtype=float))
                    for point in plane["points"]
                ],
            }
        return CachedTime(Path(path), step, time, row, plane_row)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def discover_time_caches(cache_dir: Path) -> dict[int, Path]:
    """Index cache files by solver step without trusting their contents yet."""
    cache_dir = Path(cache_dir)
    if not cache_dir.is_dir():
        return {}
    result = {}
    for path in cache_dir.glob("Nstep_*_Time_*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            result[int(payload["Nstep"])] = path
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return result
