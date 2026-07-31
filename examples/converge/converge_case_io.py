"""Input lifecycle for grid/time convergence post-processing.

Static DATA_bin is opened once.  Each dynamic directory is read, processed,
and explicitly released before the next directory is touched.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from mpcns_post import MPCNSCase


_STEP_TIME = re.compile(
    r"^Step_(?P<step>\d+)_Time_(?P<time>[+-]?[\d.]+(?:[Ee][+-]?\d+)?)$"
)


@dataclass(frozen=True)
class TimeDirectory:
    """One dynamic output directory and its time metadata."""

    path: Path
    step: int | None = None
    time: float | None = None


def open_static_case(data_dir: Path) -> MPCNSCase:
    """Open DATA_DIR/DATA_bin without loading any flow-field file."""
    data_dir = Path(data_dir)
    case = MPCNSCase.open(data_dir / "DATA_bin")
    print("Read static MPCNS data:", data_dir / "DATA_bin")
    return case


def discover_time_directories(data_dir: Path) -> list[TimeDirectory]:
    """Return archive outputs ordered by solver step.

    Expected layout is DATA_DIR/DATA_archive/Step_000357000_Time_1.140441e+01/
    containing flow_field####.bin files.
    """
    archive = Path(data_dir) / "DATA_archive"
    if not archive.is_dir():
        return []
    result = []
    for path in archive.iterdir():
        match = _STEP_TIME.match(path.name)
        if path.is_dir() and match:
            result.append(TimeDirectory(
                path=path,
                step=int(match["step"]),
                time=float(match["time"]),
            ))
    return sorted(result, key=lambda item: (item.step, item.time))


def select_inputs(data_dir: Path) -> list[TimeDirectory]:
    """Prefer DATA_archive; fall back to the single current DATA directory."""
    archived = discover_time_directories(data_dir)
    if archived:
        return archived
    current = Path(data_dir) / "DATA"
    if not current.is_dir():
        raise FileNotFoundError(
            "Expected either DATA_archive/Step_*_Time_* directories or DATA_DIR/DATA"
        )
    return [TimeDirectory(path=current)]


def load_time(case: MPCNSCase, dynamic_dir: Path) -> tuple[int, float]:
    """Read and assemble exactly one dynamic output directory."""
    restarts = case.read_latest_restart(data_dir=dynamic_dir)
    case.assemble_dynamic_fields(restarts)
    return int(restarts[0].step), float(restarts[0].time)


def release_time(case: MPCNSCase) -> None:
    """Drop restart/dynamic arrays and all derived arrays cached from them."""
    case._latest = None  # The public API intentionally has no unload method yet.
    case._dynamic = None
    case._species_cache.clear()
    case.derived.clear_cache()
