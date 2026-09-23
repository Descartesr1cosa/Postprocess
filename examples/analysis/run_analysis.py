"""Run modular MPCNS time-series analysis.

Edit ``DATA_DIR`` and the settings below, then run from the repository root:

    python examples/analysis/run_analysis.py

Every entry in ``OPERATIONS`` is independent.  Comment out exactly one entry
to disable that post-processing calculation without changing any other code.
"""

from __future__ import annotations

import os
from pathlib import Path

from analysis_cache import discover_time_caches, read_time_cache, write_time_cache
from analysis_case_io import discover_time_directories, load_time, open_static_case, release_time, select_inputs
from analysis_na import (
    make_na_asymmetry_operation,
    make_na_boundary_budget_operation,
    make_na_inventory_operation,
    make_na_source_operation,
    make_na_sphere_flux_operation,
    make_na_tail_plane_flux_operation,
)
from analysis_output import write_scalar_outputs
from analysis_regions import make_na_region_operation
from analysis_volume import make_node_volume_operation
from analysis_structure import make_global_structure_operation


# DATA_DIR contains DATA_bin plus DATA_archive/Step_*_Time_* (recommended),
# or one current DATA directory.
DATA_DIR = Path(r"E:\\2_ClassFiles\\x2025\\Autumn\\Mercury\\python\\999_Post\\DATA\\out56-Na0")
ANALYSIS_DIR = DATA_DIR / "tecplot_output" / "analysis"
SCALAR_DIR = ANALYSIS_DIR / "scalars"
VOLUME_DIR = ANALYSIS_DIR / "volume"      # One connected 3-D <time>.plt per time.
CACHE_DIR = ANALYSIS_DIR / "cache"        # Restartable per-time JSON records.
# Leave as None for all available times.  The environment override below is
# convenient for a low-cost smoke test: MPCNS_ANALYSIS_MAX_TIME_SAMPLES=1.
MAX_TIME_SAMPLES = None


# ---------------------------------------------------------------------------
# Global-structure settings
# ---------------------------------------------------------------------------
# MSO +x is subsolar.  The physical wall near x=1 R_M is deliberately omitted.
SUBSOLAR_TUBE_RADIUS_RM = 0.15
MIN_X_RM = 1.02
MAX_X_RM = 8.0
LOCAL_AVERAGE_CELLS = 5


# ---------------------------------------------------------------------------
# Na+ budget settings
# ---------------------------------------------------------------------------
# A boundary Face at r <= this value belongs to Mercury's physical surface.
# Inspect the mesh first if its wall is not at 1 R_M.
SURFACE_RADIUS_MAX_RM = 1.05

# Freely edit this tuple.  Every radius receives net/outward/inward columns.
NA_SPHERE_RADII_RM = (1.5, 1.9, 5.0)
NA_SPHERE_SAMPLE_COUNT = 4096

# Circular tail-plane transport, with positive meaning anti-sunward (-x).
TAIL_PLANE_X_RM = -5.0
TAIL_PLANE_RADIUS_RM = 5.0
TAIL_PLANE_SAMPLE_COUNT = 16_384

# These are explicit geometry-based attribution regions, not a field-line
# classifier.  Adjust their R_M bounds after inspecting each case's slices.
NA_REGIONS = (
    {"name": "near_surface", "r": (1.0, 1.5)},
    {"name": "north_cusp", "x": (-1.0, 3.0), "z": (1.0, 4.0), "r": (1.2, 4.5)},
    {"name": "south_cusp", "x": (-1.0, 3.0), "z": (-4.0, -1.0), "r": (1.2, 4.5)},
    {"name": "north_plasma_mantle", "x": (-8.0, -1.0), "z": (1.0, 5.0)},
    {"name": "south_plasma_mantle", "x": (-8.0, -1.0), "z": (-5.0, -1.0)},
    {"name": "plasma_sheet", "x": (-8.0, -1.0), "y": (-3.0, 3.0), "z": (-1.0, 1.0)},
    {"name": "tail", "x": (-8.0, -2.0), "r": (1.0, 7.0)},
    {"name": "dayside_mp_magnetosheath_proxy", "x": (1.0, 8.0), "r": (1.5, 8.0)},
)


# ---------------------------------------------------------------------------
# Switchboard: comment any one line to turn that operation off.
# ---------------------------------------------------------------------------
OPERATIONS = [
    make_global_structure_operation(
        tube_radius_rm=SUBSOLAR_TUBE_RADIUS_RM, min_x_rm=MIN_X_RM,
        max_x_rm=MAX_X_RM, neighbours=LOCAL_AVERAGE_CELLS,
    ),
    make_na_inventory_operation(),
    make_na_source_operation(),
    make_na_boundary_budget_operation(surface_radius_max_rm=SURFACE_RADIUS_MAX_RM),
    make_na_sphere_flux_operation(
        radii_rm=NA_SPHERE_RADII_RM, sphere_sample_count=NA_SPHERE_SAMPLE_COUNT,
    ),
    make_na_tail_plane_flux_operation(
        x_rm=TAIL_PLANE_X_RM, radius_rm=TAIL_PLANE_RADIUS_RM,
        sample_count=TAIL_PLANE_SAMPLE_COUNT,
    ),
    make_na_region_operation(regions=NA_REGIONS),
    make_na_asymmetry_operation(),
    make_node_volume_operation(output_dir=VOLUME_DIR),
]


def _configuration() -> dict:
    """Everything that changes an operation's cached result."""
    return {
        "cache_schema_note": "Changing enabled operations/settings recomputes affected time steps.",
        "operations": [
            {"name": operation.name, "configuration": operation.configuration}
            for operation in OPERATIONS
        ],
    }


def _quantity_units() -> dict[str, str]:
    units = {}
    for operation in OPERATIONS:
        duplicate = set(units) & set(operation.units)
        if duplicate:
            raise ValueError(f"Duplicate output quantity names: {sorted(duplicate)}")
        units.update(operation.units)
    return units


def main() -> None:
    if not OPERATIONS:
        raise ValueError("OPERATIONS is empty; enable at least one analysis operation")
    # Create all three product classes up front, including the 3-D Tecplot
    # destination, so a run has an obvious and stable analysis layout.
    VOLUME_DIR.mkdir(parents=True, exist_ok=True)
    configuration = _configuration()
    units = _quantity_units()
    expected_columns = {"time", "Nstep", *units}
    cache_paths = discover_time_caches(CACHE_DIR)
    archive_by_step = {item.step: item for item in discover_time_directories(DATA_DIR)}
    if not archive_by_step:
        # Preserve the single-current-DATA workflow when no archive exists.
        current = select_inputs(DATA_DIR)
        archive_by_step = {
            item.step if item.step is not None else -(index + 1): item
            for index, item in enumerate(current)
        }
    keys = sorted(set(archive_by_step) | set(cache_paths))
    if not keys:
        raise FileNotFoundError("No archive time directories or analysis cache files were found")
    configured_limit = os.environ.get("MPCNS_ANALYSIS_MAX_TIME_SAMPLES")
    limit = MAX_TIME_SAMPLES if configured_limit is None else int(configured_limit)
    if limit is not None:
        if limit < 1:
            raise ValueError("MAX_TIME_SAMPLES must be positive or None")
        keys = keys[:limit]
        print(f"Processing only the first {len(keys)} time sample(s)")

    rows = []
    case = None
    state: dict = {}
    prepared = False
    for key in keys:
        cached = read_time_cache(cache_paths[key], configuration=configuration) if key in cache_paths else None
        if cached is not None and set(cached.row) == expected_columns:
            rows.append(cached.row)
            print(f"Reused cached step={cached.step} time={cached.time:.6e}: {cached.path.name}")
            continue
        source = archive_by_step.get(key)
        if source is None:
            raise RuntimeError(f"Cache for step {key} is incompatible, and its dynamic output is unavailable")
        try:
            if case is None:
                case = open_static_case(DATA_DIR)
            step, time = load_time(case, source.path)
            if not prepared:
                for operation in OPERATIONS:
                    if operation.prepare is not None:
                        operation.prepare(case, state)
                prepared = True
            row = {
                "time": time if source.time is None else source.time,
                "Nstep": step if source.step is None else source.step,
            }
            for operation in OPERATIONS:
                values = operation.calculate(case, state)
                duplicate = set(row) & set(values)
                if duplicate:
                    raise ValueError(f"{operation.name} produced duplicate columns: {sorted(duplicate)}")
                row.update(values)
            rows.append(row)
            print(f"Processed step={row['Nstep']:.0f} time={row['time']:.6e}")
            print("  Cached:", write_time_cache(CACHE_DIR, row, configuration=configuration))
        finally:
            if case is not None:
                release_time(case)

    rows.sort(key=lambda row: (row["Nstep"], row["time"]))
    dat_path, json_path = write_scalar_outputs(rows, SCALAR_DIR, units=units)
    print("Written:", dat_path)
    print("Written:", json_path)
    print("3-D Tecplot directory:", VOLUME_DIR)


if __name__ == "__main__":
    main()
