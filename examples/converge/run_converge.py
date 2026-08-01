"""Run subsolar magnetopause/bow-shock convergence processing.

Edit DATA_DIR and settings below, then run from the repository root:
    python examples/converge/run_converge.py
"""

from __future__ import annotations

from pathlib import Path

from converge_case_io import load_time, open_static_case, release_time, select_inputs
from converge_core import dynamic_quantities, make_subsolar_samples, prepare_static, reconstruct_additive_B_cell
from converge_output import write_outputs
from converge_plane_output import write_plane_topology
from converge_plane_topology import find_plane_xo_points
from converge_positions import bow_shock_x, magnetopause_x


# DATA_DIR contains DATA_bin plus either DATA_archive/Step_*_Time_* or DATA.
DATA_DIR = Path(r"path\to\your\DATA_DIR")
# DATA_DIR = Path(r"E:\\2_ClassFiles\\x2025\\Autumn\\Mercury\\python\\999_Post\\DATA\\out56")
# Same output convention as examples/post: DATA_DIR/tecplot_output.
OUTPUT_DIR = DATA_DIR / "tecplot_output"

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

# Independent cell-centred X/O topology output.  The selected plane is a slab
# because no interpolation is performed; choose a tolerance matching your
# local cell spacing (use 0 for an exactly represented symmetry plane).
RUN_PLANE_TOPOLOGY = True
PLANE_NORMAL_AXIS = "y"       # one of: x, y, z
PLANE_VALUE_RM = 0.0
PLANE_TOLERANCE_RM = 1.0e-8
# Merge repeated adjacent quadrilateral contours of one physical zero.
PLANE_MERGE_RADIUS_IN_SPACINGS = 3.0


def main() -> None:
    case = open_static_case(DATA_DIR)
    rows = []
    plane_rows = []
    # This is static, and is therefore deliberately retained through all times.
    additive_b_nd = None
    static = None
    for source in select_inputs(DATA_DIR):
        try:
            step, time = load_time(case, source.path)
            if additive_b_nd is None:
                additive_b_nd = reconstruct_additive_B_cell(case)
                static = prepare_static(case)
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
            rows.append(row)
            if RUN_PLANE_TOPOLOGY:
                points = find_plane_xo_points(
                    case, quantities["B_total_T"], quantities["fluid_mask"],
                    normal_axis=PLANE_NORMAL_AXIS, value_rm=PLANE_VALUE_RM,
                    tolerance_rm=PLANE_TOLERANCE_RM,
                    merge_radius_in_spacings=PLANE_MERGE_RADIUS_IN_SPACINGS,
                )
                plane_rows.append({"time": row["time"], "Nstep": row["Nstep"], "points": points})
                print(
                    "  Plane X/O points: X={}, O={}".format(
                        sum(point.kind == "X" for point in points),
                        sum(point.kind == "O" for point in points),
                    )
                )
            print("Processed step={Nstep} time={time:.6e}: mp={magnetopause_x_RM:.4f}, bs={bow_shock_x_RM:.4f}".format(**row))
        finally:
            # Never retain the current flow_field data while advancing time.
            release_time(case)
    dat_path, json_path = write_outputs(
        rows, OUTPUT_DIR,
        plane_topology_rows=plane_rows if RUN_PLANE_TOPOLOGY else None,
    )
    print("Written:", dat_path)
    print("Written:", json_path)
    if RUN_PLANE_TOPOLOGY:
        dat_path, detail_path = write_plane_topology(plane_rows, OUTPUT_DIR)
        print("Written:", dat_path)
        print("Written:", detail_path)


if __name__ == "__main__":
    main()
