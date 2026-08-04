"""Run subsolar magnetopause/bow-shock convergence processing.

Edit DATA_DIR and settings below, then run from the repository root:
    python examples/converge/run_converge.py
"""

from __future__ import annotations

from pathlib import Path

from converge_case_io import discover_time_directories, load_time, open_static_case, release_time, select_inputs
from converge_core import dynamic_quantities, make_subsolar_samples, prepare_static, reconstruct_additive_B_cell
from converge_output import write_outputs
from converge_na_statistics import build_sphere_flux_samplers, compute_na_statistics
from converge_plane_output import write_plane_topology
from converge_plane_topology import find_plane_xo_points
from converge_positions import bow_shock_x, magnetopause_x
from converge_time_cache import discover_time_caches, read_time_cache, write_time_cache


# DATA_DIR contains DATA_bin plus either DATA_archive/Step_*_Time_* or DATA.
DATA_DIR = Path(r"path\to\your\DATA_DIR")
# DATA_DIR = Path(r"E:\\2_ClassFiles\\x2025\\Autumn\\Mercury\\python\\999_Post\\DATA\\out56")
# Same output convention as examples/post: DATA_DIR/tecplot_output.
OUTPUT_DIR = DATA_DIR / "tecplot_output"
TEMP_POST_DATA_DIR = DATA_DIR / "temp_post_data"

# MSO +x is the subsolar direction.  Exclude the physical wall at x=1 R_M.
SUBSOLAR_TUBE_RADIUS_RM = 0.15
MIN_X_RM = 1.02
MAX_X_RM = 8.0
LOCAL_AVERAGE_CELLS = 5  # Set 3--5 according to the local mesh resolution.

# Add any future scalar Q processor here; converge_output.py automatically
# creates its Tecplot column and mean/standard-deviation JSON entry.
MEASUREMENTS = {
    "magnetopause_x_RM": magnetopause_x,
    "bow_shock_x_RM": bow_shock_x,
}
MEASUREMENT_UNITS = {
    "magnetopause_x_RM": "Mercury radii (R_M)",
    "bow_shock_x_RM": "Mercury radii (R_M)",
}

# Whole-fluid-domain Na+ number-density statistics and net outward particle
# flux through virtual spheres centred at Mercury.
RUN_NA_STATISTICS = True
NA_FLUX_RADII_RM = (1.5, 1.9, 5.0)
NA_SPHERE_SAMPLE_COUNT = 4096


def _na_flux_variable_name(radius_rm: float) -> str:
    label = f"{radius_rm:g}".replace("-", "m").replace(".", "p")
    return f"Na_flux_outward_r{label}_RM_particles_s"


MEASUREMENT_UNITS.update({
    "Na_number_density_domain_mean_cm3": "cm^-3",
    "Na_number_density_domain_std_cm3": "cm^-3",
    **{_na_flux_variable_name(radius): "particles/s" for radius in NA_FLUX_RADII_RM},
})

# Independent cell-centred X/O topology output.  The selected plane is a slab
# because no interpolation is performed; choose a tolerance matching your
# local cell spacing (use 0 for an exactly represented symmetry plane).
RUN_PLANE_TOPOLOGY = True
PLANE_NORMAL_AXIS = "y"       # one of: x, y, z
PLANE_VALUE_RM = 0.0
PLANE_TOLERANCE_RM = 1.0e-8
# Merge repeated adjacent quadrilateral contours of one physical zero.
PLANE_MERGE_RADIUS_IN_SPACINGS = 3.0
# "cell_center": existing grid Cell centre; "interpolated": local affine
# B_t=0 coordinate in the accepted four-Cell topology contour.
PLANE_POINT_POSITION_MODE = "interpolated"


def cache_configuration() -> dict:
    """Settings that affect one cached time step's values or topology."""
    return {
        "subsolar": {
            "tube_radius_rm": SUBSOLAR_TUBE_RADIUS_RM,
            "min_x_rm": MIN_X_RM,
            "max_x_rm": MAX_X_RM,
            "local_average_cells": LOCAL_AVERAGE_CELLS,
        },
        "na_statistics": {
            "enabled": RUN_NA_STATISTICS,
            "flux_radii_rm": list(NA_FLUX_RADII_RM),
            "sphere_sample_count": NA_SPHERE_SAMPLE_COUNT,
        },
        "plane_topology": {
            "enabled": RUN_PLANE_TOPOLOGY,
            "normal_axis": PLANE_NORMAL_AXIS,
            "value_rm": PLANE_VALUE_RM,
            "tolerance_rm": PLANE_TOLERANCE_RM,
            "merge_radius_in_spacings": PLANE_MERGE_RADIUS_IN_SPACINGS,
            "position_mode": PLANE_POINT_POSITION_MODE,
        },
    }


def main() -> None:
    rows = []
    plane_rows = []
    configuration = cache_configuration()
    cache_paths = discover_time_caches(TEMP_POST_DATA_DIR)
    archive_by_step = {item.step: item for item in discover_time_directories(DATA_DIR)}
    if not archive_by_step and (DATA_DIR / "DATA").is_dir():
        # Retain the one-current-DATA workflow when no history archive exists.
        current = select_inputs(DATA_DIR)
        archive_by_step = {item.step: item for item in current if item.step is not None}
        if not archive_by_step:
            # Current DATA has no known Nstep until it is read, so it cannot be
            # looked up in cache first; it remains a normal compute input.
            archive_by_step = {-(index + 1): item for index, item in enumerate(current)}
    all_keys = sorted(set(archive_by_step) | set(cache_paths))
    if not all_keys:
        raise FileNotFoundError("No DATA_archive time directories or temp_post_data cache files were found")

    case = None
    # This is static, and is therefore deliberately retained through all times.
    additive_b_nd = None
    static = None
    na_sphere_samplers = None
    for key in all_keys:
        cached = read_time_cache(cache_paths[key], configuration=configuration) if key in cache_paths else None
        if cached is not None:
            rows.append(cached.row)
            if RUN_PLANE_TOPOLOGY:
                if cached.plane_row is None:
                    raise RuntimeError(f"Cached step {cached.step} lacks enabled plane-topology data: {cached.path}")
                plane_rows.append(cached.plane_row)
            print(f"Reused cached step={cached.step} time={cached.time:.6e}: {cached.path.name}")
            continue
        source = archive_by_step.get(key)
        if source is None:
            raise RuntimeError(
                f"Cache for step {key} is incompatible/corrupt and its DATA_archive directory is unavailable"
            )
        try:
            if case is None:
                case = open_static_case(DATA_DIR)
            step, time = load_time(case, source.path)
            if additive_b_nd is None:
                additive_b_nd = reconstruct_additive_B_cell(case)
                static = prepare_static(case)
                if RUN_NA_STATISTICS:
                    na_sphere_samplers = build_sphere_flux_samplers(
                        case, static["fluid_mask"], NA_FLUX_RADII_RM,
                        sphere_sample_count=NA_SPHERE_SAMPLE_COUNT,
                    )
            quantities = dynamic_quantities(case, additive_b_nd)
            samples = make_subsolar_samples(
                static, quantities,
                transverse_radius_rm=SUBSOLAR_TUBE_RADIUS_RM,
                min_x_rm=MIN_X_RM, max_x_rm=MAX_X_RM,
                neighbours=LOCAL_AVERAGE_CELLS,
            )
            row = {
                "time": time if source.time is None else source.time,
                "Nstep": step if source.step is None else source.step,
            }
            row.update({name: calculator(samples) for name, calculator in MEASUREMENTS.items()})
            if RUN_NA_STATISTICS:
                na_values = compute_na_statistics(case, quantities["fluid_mask"], na_sphere_samplers)
                row.update(na_values)
            rows.append(row)
            if RUN_PLANE_TOPOLOGY:
                points = find_plane_xo_points(
                    case, quantities["B_total_T"], quantities["fluid_mask"],
                    normal_axis=PLANE_NORMAL_AXIS, value_rm=PLANE_VALUE_RM,
                    tolerance_rm=PLANE_TOLERANCE_RM,
                    merge_radius_in_spacings=PLANE_MERGE_RADIUS_IN_SPACINGS,
                    position_mode=PLANE_POINT_POSITION_MODE,
                )
                plane_rows.append({
                    "time": row["time"], "Nstep": row["Nstep"], "points": points,
                    "coordinate_mode": PLANE_POINT_POSITION_MODE,
                })
                print(
                    "  Plane X/O points: X={}, O={}".format(
                        sum(point.kind == "X" for point in points),
                        sum(point.kind == "O" for point in points),
                    )
                )
            print("Processed step={Nstep} time={time:.6e}: mp={magnetopause_x_RM:.4f}, bs={bow_shock_x_RM:.4f}".format(**row))
            cache_path = write_time_cache(
                TEMP_POST_DATA_DIR, row,
                plane_rows[-1] if RUN_PLANE_TOPOLOGY else None,
                configuration=configuration,
            )
            print("  Cached:", cache_path)
        finally:
            # Never retain the current flow_field data while advancing time.
            if case is not None:
                release_time(case)
    rows.sort(key=lambda row: (row["Nstep"], row["time"]))
    plane_rows.sort(key=lambda row: (row["Nstep"], row["time"]))
    dat_path, json_path = write_outputs(
        rows, OUTPUT_DIR,
        plane_topology_rows=plane_rows if RUN_PLANE_TOPOLOGY else None,
        quantity_units=MEASUREMENT_UNITS,
    )
    print("Written:", dat_path)
    print("Written:", json_path)
    if RUN_PLANE_TOPOLOGY:
        dat_path, detail_path = write_plane_topology(plane_rows, OUTPUT_DIR)
        print("Written:", dat_path)
        print("Written:", detail_path)


if __name__ == "__main__":
    main()
