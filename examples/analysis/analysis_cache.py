"""Small, configuration-aware checkpoint files for resumable analysis."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


CACHE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CachedTime:
    path: Path
    step: int
    time: float
    row: dict[str, float]


def _canonical_configuration(configuration: dict) -> dict:
    """Normalize tuples/NumPy scalars exactly as they are stored in JSON."""
    return json.loads(json.dumps(_json_value(configuration), ensure_ascii=False, sort_keys=True))


def _filename(step: int, time: float) -> str:
    return f"Nstep_{step:09d}_Time_{time:.12e}.json"


def _json_value(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    return value


def write_time_cache(cache_dir: Path, row: dict[str, float], *, configuration: dict) -> Path:
    """Write one independent per-time checkpoint."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    step, time = int(row["Nstep"]), float(row["time"])
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "configuration": _canonical_configuration(configuration),
        "Nstep": step,
        "time": time,
        "row": _json_value(row),
    }
    path = cache_dir / _filename(step, time)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def read_time_cache(path: Path, *, configuration: dict) -> CachedTime | None:
    """Read a matching cache; incompatible or incomplete caches are ignored."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload["schema_version"] != CACHE_SCHEMA_VERSION or payload["configuration"] != _canonical_configuration(configuration):
            return None
        step, time = int(payload["Nstep"]), float(payload["time"])
        row = {name: float("nan") if value is None else float(value) for name, value in payload["row"].items()}
        if int(row["Nstep"]) != step or not np.isclose(row["time"], time, rtol=0.0, atol=1e-12):
            return None
        return CachedTime(Path(path), step, time, row)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def discover_time_caches(cache_dir: Path) -> dict[int, Path]:
    """Index checkpoints by step without trusting their contents yet."""
    if not Path(cache_dir).is_dir():
        return {}
    result = {}
    for path in Path(cache_dir).glob("Nstep_*_Time_*.json"):
        try:
            result[int(json.loads(path.read_text(encoding="utf-8"))["Nstep"])] = path
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return result
