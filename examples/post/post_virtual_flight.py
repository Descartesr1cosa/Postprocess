"""Virtual MESSENGER flight export; MAG file reading stays in messenger_mso_data."""

# MESSENGER positions are in km, whereas MPCNS coordinates are in R_M.
MERCURY_RADIUS_KM = 2440.0

VIRTUAL_FLIGHT_IMPLEMENTATION = r'''
+# ============================================================
# 18. Virtual flight along the MESSENGER MSO trajectory (Cell data)
# ============================================================
#
# Set RUN_VIRTUAL_FLIGHT = True when a virtual flight is wanted.  The spacecraft
# position in the MAG files and the MPCNS calculation are both in MSO.  The
# fields below are sampled from the *nearest Cell center* (not reconstructed
# Nodes), which is the usual choice for a Cell-centred simulation field.

RUN_VIRTUAL_FLIGHT = globals().get("RUN_VIRTUAL_FLIGHT", True)
VIRTUAL_FLIGHT_OUTPUT_PATH = globals().get(
    "VIRTUAL_FLIGHT_OUTPUT_PATH",
    OUTPUT_DIR / "06_virtual_flight_cell.plt",
)

# Add/remove names here to choose the simulation fields to export.  Valid names
# are the keys in CELL_ARRAYS printed below.  Scalar arrays make one PLT
# variable; (N, 3) vectors automatically make _x, _y and _z variables.
VIRTUAL_FLIGHT_FIELDS = globals().get("VIRTUAL_FLIGHT_FIELDS", [
    "B_total_nT",
    "H_number_density_cm3",
    "H_pressure_nPa",
    "H_temperature_K",
    "Na_number_density_cm3",
    "Na_pressure_nPa",
    "Na_temperature_K",
])

# These settings are passed to messenger_mso_data.py.  UTC uses ISO 8601; None
# means the earliest/latest available MAG time.  Set the average window to None
# to use raw MAG samples, and control their spacing with SAMPLE_EVERY_SECONDS.
VIRTUAL_FLIGHT_TIME_START = globals().get("VIRTUAL_FLIGHT_TIME_START", "2008-10-06T07:00:00")
VIRTUAL_FLIGHT_TIME_STOP = globals().get("VIRTUAL_FLIGHT_TIME_STOP", "2008-10-06T10:00:00")
VIRTUAL_FLIGHT_AVERAGE_WINDOW_SECONDS = globals().get("VIRTUAL_FLIGHT_AVERAGE_WINDOW_SECONDS", None)
VIRTUAL_FLIGHT_SAMPLE_EVERY_SECONDS = globals().get("VIRTUAL_FLIGHT_SAMPLE_EVERY_SECONDS", 15.0)


def write_virtual_messenger_flight():
    """Sample selected Cell arrays at MESSENGER's MSO positions and write PLT."""
    from scipy.spatial import cKDTree
    from messenger_mso_data import load_messenger_data

    missing = [name for name in VIRTUAL_FLIGHT_FIELDS if name not in CELL_ARRAYS]
    if missing:
        raise KeyError(
            "Unknown VIRTUAL_FLIGHT_FIELDS: " + ", ".join(missing)
            + "\nAvailable CELL_ARRAYS: " + ", ".join(CELL_ARRAYS)
        )

    messenger = load_messenger_data(
        VIRTUAL_FLIGHT_TIME_START,
        VIRTUAL_FLIGHT_TIME_STOP,
        data_dir=MESSENGER_DATA_DIR,
        average_window_seconds=VIRTUAL_FLIGHT_AVERAGE_WINDOW_SECONDS,
        sample_every_seconds=VIRTUAL_FLIGHT_SAMPLE_EVERY_SECONDS,
    )

    # MAG positions are km; MPCNS Cell coordinates are normalized by R_M.
    messenger_xyz_RM = messenger[:, 4:7] / MERCURY_RADIUS_KM
    tree = cKDTree(case.cells.coordinates)
    distance_RM, cell_indices = tree.query(messenger_xyz_RM, k=1)

    table_fields = {
        "utc_unix_s": messenger[:, 0],
        "utc_year": messenger[:, 1],
        "utc_day_of_year": messenger[:, 2],
        "utc_seconds_of_day": messenger[:, 3],
        "x_mso_RM": messenger_xyz_RM[:, 0],
        "y_mso_RM": messenger_xyz_RM[:, 1],
        "z_mso_RM": messenger_xyz_RM[:, 2],
        "R_mso_RM": np.linalg.norm(messenger_xyz_RM, axis=1),
        "nearest_cell_distance_RM": distance_RM,
        "cell_global_id": case.cells.global_ids[cell_indices],
        "messenger_Bx_mso_nT": messenger[:, 7],
        "messenger_By_mso_nT": messenger[:, 8],
        "messenger_Bz_mso_nT": messenger[:, 9],
    }
    for field_name in VIRTUAL_FLIGHT_FIELDS:
        values = np.asarray(CELL_ARRAYS[field_name])[cell_indices]
        if values.ndim == 1:
            table_fields[f"sim_{field_name}"] = values
        elif values.ndim == 2 and values.shape[1] == 3:
            for component, label in enumerate(("x", "y", "z")):
                table_fields[f"sim_{field_name}_{label}"] = values[:, component]
        else:
            raise ValueError(
                f"{field_name} must be a scalar or 3-vector Cell array, got {values.shape}"
            )

    # Tecplot needs ordered 3-D arrays.  A trajectory is represented as I=N,
    # J=K=1.  UTC is retained both as Unix seconds and as year/day/seconds-day
    # so sub-second values and a human-readable calendar representation survive.
    zone_values = {
        name: np.asarray(values, dtype=np.float64).reshape((-1, 1, 1))
        for name, values in table_fields.items()
    }
    written = write_tecplot_binary(
        VIRTUAL_FLIGHT_OUTPUT_PATH,
        title=(
            "MESSENGER MSO virtual flight: nearest MPCNS Cell values; "
            "time is UTC Unix seconds plus year/day/seconds-of-day"
        ),
        variable_names=tuple(zone_values),
        zones=[TecplotZone(
            name="MESSENGER_MSO_virtual_flight",
            physics="Fluid",
            values=zone_values,
        )],
        solution_time=float(case.latest_restart[0].time),
    )
    info = inspect_tecplot_binary(written)
    print("Written virtual-flight PLT:", info.path)
    print("  trajectory samples:", messenger.shape[0])
    print("  selected Cell fields:", ", ".join(VIRTUAL_FLIGHT_FIELDS))


if RUN_VIRTUAL_FLIGHT:
    print()
    print("Available Cell fields for virtual flight:")
    print("  " + "\n  ".join(CELL_ARRAYS))
    write_virtual_messenger_flight()

'''

from pathlib import Path

from mpcns_post.tecplot import (
    TecplotZone,
    inspect_tecplot_binary,
    write_tecplot_binary,
)


FIELDS = [
    "B_total_nT", "H_number_density_cm3", "H_pressure_nPa", "H_temperature_K",
    "Na_number_density_cm3", "Na_pressure_nPa", "Na_temperature_K",
]
TIME_START = "2008-10-06T07:00:00"
TIME_STOP = "2008-10-06T10:00:00"
AVERAGE_WINDOW_SECONDS = None
SAMPLE_EVERY_SECONDS = 15.0


def export_virtual_flight(
    data: dict,
    *,
    messenger_data_dir: Path,
) -> None:
    """Configure and call the Cell-centred virtual-flight writer."""
    scope = dict(data)
    scope["RUN_VIRTUAL_FLIGHT"] = True
    scope["MERCURY_RADIUS_KM"] = MERCURY_RADIUS_KM
    # The top-level switchboard owns this path so a run has one source of truth.
    scope["MESSENGER_DATA_DIR"] = Path(messenger_data_dir)
    scope["VIRTUAL_FLIGHT_FIELDS"] = FIELDS
    scope["VIRTUAL_FLIGHT_TIME_START"] = TIME_START
    scope["VIRTUAL_FLIGHT_TIME_STOP"] = TIME_STOP
    scope["VIRTUAL_FLIGHT_AVERAGE_WINDOW_SECONDS"] = AVERAGE_WINDOW_SECONDS
    scope["VIRTUAL_FLIGHT_SAMPLE_EVERY_SECONDS"] = SAMPLE_EVERY_SECONDS
    scope["VIRTUAL_FLIGHT_OUTPUT_PATH"] = Path(data["OUTPUT_DIR"]) / "06_virtual_flight_cell.plt"
    scope["TecplotZone"] = TecplotZone
    scope["inspect_tecplot_binary"] = inspect_tecplot_binary
    scope["write_tecplot_binary"] = write_tecplot_binary
    exec(VIRTUAL_FLIGHT_IMPLEMENTATION.replace("\n+", "\n"), scope)
