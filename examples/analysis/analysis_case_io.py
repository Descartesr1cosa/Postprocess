"""Input lifecycle for time-series MPCNS analysis.

Only the static ``DATA_bin`` mesh is retained.  A dynamic output is assembled,
analysed, and explicitly released before the next output is opened.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from mpcns_post import MPCNSCase


_STEP_TIME = re.compile(r"^Step_(?P<step>\d+)_Time_(?P<time>[+-]?[\d.]+(?:[Ee][+-]?\d+)?)$")


@dataclass(frozen=True)
class TimeDirectory:
    """A solver output directory with metadata parsed from its name."""

    path: Path
    step: int | None = None
    time: float | None = None


def open_static_case(data_dir: Path) -> MPCNSCase:
    """Open the static mesh and constants without loading a flow field."""
    path = Path(data_dir) / "DATA_bin"
    case = MPCNSCase.open(path)
    print("Read static MPCNS data:", path)
    return case


def discover_time_directories(data_dir: Path) -> list[TimeDirectory]:
    """Find archived ``Step_*_Time_*`` outputs in solver-step order."""
    archive = Path(data_dir) / "DATA_archive"
    if not archive.is_dir():
        return []
    items = []
    for path in archive.iterdir():
        match = _STEP_TIME.match(path.name)
        if path.is_dir() and match:
            items.append(TimeDirectory(path, int(match["step"]), float(match["time"])))
    return sorted(items, key=lambda item: (item.step, item.time))


def select_inputs(data_dir: Path) -> list[TimeDirectory]:
    """Prefer archive history; otherwise process the current ``DATA`` output."""
    archived = discover_time_directories(data_dir)
    if archived:
        return archived
    current = Path(data_dir) / "DATA"
    if not current.is_dir():
        raise FileNotFoundError("Expected DATA_archive/Step_*_Time_* or DATA")
    return [TimeDirectory(current)]


def load_time(case: MPCNSCase, dynamic_dir: Path) -> tuple[int, float]:
    """Load and assemble exactly one dynamic solver output."""
    restarts = case.read_latest_restart(data_dir=dynamic_dir)
    case.assemble_dynamic_fields(restarts)
    return int(restarts[0].step), float(restarts[0].time)


def release_time(case: MPCNSCase) -> None:
    """Release restart arrays and all derived arrays before advancing time."""
    # The current public API has no unload method; clear all dynamic ownership
    # points together so a long archive does not retain old flow fields.
    case._latest = None
    case._dynamic = None
    case._species_cache.clear()
    case.derived.clear_cache()
