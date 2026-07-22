"""Export Na+ altitude profiles using already calculated post_core data."""

NA_ALTITUDE_IMPLEMENTATION = r'''
+# ============================================================
# 17. Export MSO local-time Na+ altitude profiles as ASCII
#
# For each local-time region, one file is written:
#
#     dawn.dat
#     subsolar.dat
#     dusk.dat
#
# Each file contains:
#
#     1. simulated Na+ density;
#     2. reference density using the fitted B;
#     3. reference_1 using B - delta_B;
#     4. reference_2 using B + delta_B.
#
# Statistics:
#
#             sum_i q_i V_i
#     q_bar = -------------
#               sum_i V_i
#
# The sum is taken over all valid Fluid Cells inside:
#
#     1. one MSO local-time window;
#     2. one altitude bin.
#
# All latitudes are included.
# ============================================================


# ------------------------------------------------------------
# 17.1 User settings
# ------------------------------------------------------------

PROFILE_OUTPUT_DIR = (
    OUTPUT_DIR
    / "Na_altitude_ascii"
)

if RUN_NA_ALTITUDE_PROFILES:
    PROFILE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Altitude range in km.
PROFILE_H_MIN_KM = 0.0
PROFILE_H_MAX_KM = 6000.0


# Altitude-bin width in km.
PROFILE_H_BIN_KM = 80.0


# Local-time half-width in hours.
#
#     dawn:      4.5 -- 7.5 h
#     subsolar: 10.5 -- 13.5 h
#     dusk:     16.5 -- 19.5 h
LT_WINDOW_HALF_WIDTH_HOUR = 1.5


# ------------------------------------------------------------
# Literature reference models
#
#     n_ref(H)
#       = n0 * exp(-(H - H0) / B)
#
# Units:
#
#     n0      : cm^-3
#     H0      : km
#     B       : km
#     B_error : km
#
# The B uncertainty is different for each local-time region.
# ------------------------------------------------------------

NA_REFERENCE_MODELS = {
    "dawn": {
        "n0_cm3": 0.1,
        "H0_km": 800.0,
        "B_km": 670.0,
        "B_error_km": 40.0,
    },

    "subsolar": {
        "n0_cm3": 0.1,
        "H0_km": 1100.0,
        "B_km": 460.0,
        "B_error_km": 30.0,
    },

    "dusk": {
        "n0_cm3": 0.1,
        "H0_km": 2500.0,
        "B_km": 580.0,
        "B_error_km": 40.0,
    },
}


# MSO local-time centers.
#
# MSO convention:
#
#     +X : 12 h, subsolar
#     +Y : 18 h, dusk
#     -X :  0 h, midnight
#     -Y :  6 h, dawn
MSO_LOCAL_TIME_CENTERS = {
    "dawn": 6.0,
    "subsolar": 12.0,
    "dusk": 18.0,
}


# ============================================================
# 17.2 MSO local-time utilities
# ============================================================

def compute_mso_local_time_hour(
    xyz_RM,
):
    """
    Compute MSO local time from Cartesian coordinates.

    Convention
    ----------
    +X : 12 h
    +Y : 18 h
    -X :  0 h
    -Y :  6 h

    Parameters
    ----------
    xyz_RM : ndarray, shape (N, 3)
        Cartesian MSO coordinates in Mercury radii.

    Returns
    -------
    local_time_hour : ndarray, shape (N,)
        Local time in [0, 24).
    """

    xyz_RM = np.asarray(
        xyz_RM,
        dtype=np.float64,
    )

    if (
        xyz_RM.ndim != 2
        or xyz_RM.shape[1] != 3
    ):
        raise ValueError(
            "xyz_RM must have shape (N, 3)"
        )


    # Longitude measured from +X toward +Y.
    longitude_rad = np.arctan2(
        xyz_RM[:, 1],
        xyz_RM[:, 0],
    )


    local_time_hour = np.mod(
        12.0
        + longitude_rad
        * 12.0
        / np.pi,
        24.0,
    )

    return local_time_hour


def local_time_distance_hour(
    local_time_hour,
    center_hour,
):
    """
    Compute the shortest periodic local-time distance.
    """

    local_time_hour = np.asarray(
        local_time_hour,
        dtype=np.float64,
    )

    center_hour = float(
        center_hour
    )


    distance_hour = np.abs(
        (
            local_time_hour
            - center_hour
            + 12.0
        )
        % 24.0
        - 12.0
    )

    return distance_hour


def volume_weighted_mean(
    values,
    weights,
):
    """
    Compute a volume-weighted mean.
    """

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    weights = np.asarray(
        weights,
        dtype=np.float64,
    )


    weight_sum = np.sum(
        weights
    )

    if (
        not np.isfinite(weight_sum)
        or weight_sum <= 0.0
    ):
        raise ValidationError(
            "Invalid Cell-volume sum in altitude bin"
        )


    if values.ndim == 1:

        return np.sum(
            values
            * weights
        ) / weight_sum


    if values.ndim == 2:

        return np.sum(
            values
            * weights[:, None],
            axis=0,
        ) / weight_sum


    raise ValueError(
        "values must have shape (N,) or (N,M)"
    )


# ============================================================
# 17.3 Prepare global Cell quantities
# ============================================================

PROFILE_CELL_XYZ_RM = np.asarray(
    case.cells.coordinates,
    dtype=np.float64,
)


PROFILE_CELL_RADIUS_RM = np.linalg.norm(
    PROFILE_CELL_XYZ_RM,
    axis=1,
)


# Convert one Mercury radius from nondimensional units to km.
MERCURY_RADIUS_KM = float(
    np.asarray(
        units.convert(
            np.array(
                [1.0],
                dtype=np.float64,
            ),
            quantity="length",
            unit="km",
        )
    ).reshape(-1)[0]
)


PROFILE_CELL_ALTITUDE_KM = (
    PROFILE_CELL_RADIUS_RM
    - 1.0
) * MERCURY_RADIUS_KM


PROFILE_CELL_LOCAL_TIME_HOUR = (
    compute_mso_local_time_hour(
        PROFILE_CELL_XYZ_RM
    )
)


PROFILE_CELL_NA_CM3 = np.asarray(
    Na_number_density_cm3,
    dtype=np.float64,
)


# Cell measure is used only as a relative volume weight.
PROFILE_CELL_VOLUME = np.asarray(
    case.cells.measure,
    dtype=np.float64,
).reshape(-1)


if PROFILE_CELL_VOLUME.shape != (
    case.cells.size,
):
    raise ValidationError(
        "case.cells.measure has an unexpected shape: "
        f"{PROFILE_CELL_VOLUME.shape}"
    )


if np.any(
    PROFILE_CELL_VOLUME[fluid_mask] <= 0.0
):
    raise ValidationError(
        "Non-positive Cell volume in Fluid region"
    )


PROFILE_BASE_VALID_MASK = (
    np.asarray(
        fluid_mask,
        dtype=bool,
    )
    & np.isfinite(
        PROFILE_CELL_ALTITUDE_KM
    )
    & np.isfinite(
        PROFILE_CELL_LOCAL_TIME_HOUR
    )
    & np.isfinite(
        PROFILE_CELL_NA_CM3
    )
    & np.isfinite(
        PROFILE_CELL_VOLUME
    )
    & (
        PROFILE_CELL_VOLUME
        > 0.0
    )
    & (
        PROFILE_CELL_ALTITUDE_KM
        >= PROFILE_H_MIN_KM
    )
    & (
        PROFILE_CELL_ALTITUDE_KM
        <= PROFILE_H_MAX_KM
    )
)


# ============================================================
# 17.4 Construct one local-time altitude table
# ============================================================

def build_Na_local_time_altitude_table(
    location_name,
):
    """
    Construct one volume-weighted Na+ altitude profile.

    Output columns
    --------------
    0 : x_RM
    1 : y_RM
    2 : z_RM
    3 : H_km
    4 : nNa_sim_cm-3
    5 : nNa_reference_cm-3
    6 : nNa_reference_1_cm-3, B - B_error
    7 : nNa_reference_2_cm-3, B + B_error

    The simulation and all reference profiles use exactly
    the same Cells and volume weights in every altitude bin.
    """

    if (
        location_name
        not in MSO_LOCAL_TIME_CENTERS
    ):
        raise KeyError(
            f"Unknown local-time region: {location_name}"
        )


    model = NA_REFERENCE_MODELS[
        location_name
    ]

    center_hour = float(
        MSO_LOCAL_TIME_CENTERS[
            location_name
        ]
    )

    n0_cm3 = float(
        model[
            "n0_cm3"
        ]
    )

    H0_km = float(
        model[
            "H0_km"
        ]
    )

    B_km = float(
        model[
            "B_km"
        ]
    )

    B_error_km = float(
        model[
            "B_error_km"
        ]
    )


    B_reference_1_km = (
        B_km
        - B_error_km
    )

    B_reference_2_km = (
        B_km
        + B_error_km
    )


    if n0_cm3 <= 0.0:
        raise ValueError(
            f"{location_name}: n0_cm3 must be positive"
        )

    if B_km <= 0.0:
        raise ValueError(
            f"{location_name}: B_km must be positive"
        )

    if B_error_km < 0.0:
        raise ValueError(
            f"{location_name}: B_error_km must be non-negative"
        )

    if B_reference_1_km <= 0.0:
        raise ValueError(
            f"{location_name}: B_km - B_error_km "
            "must be positive"
        )


    # Select the local-time sector.
    local_time_mask = (
        local_time_distance_hour(
            PROFILE_CELL_LOCAL_TIME_HOUR,
            center_hour,
        )
        <= LT_WINDOW_HALF_WIDTH_HOUR
    )


    region_mask = (
        PROFILE_BASE_VALID_MASK
        & local_time_mask
    )


    # Evaluate all three reference curves at each Cell altitude.
    #
    # They will subsequently be averaged using the same
    # volume weights as the simulation.
    reference_cell_cm3 = (
        n0_cm3
        * np.exp(
            -(
                PROFILE_CELL_ALTITUDE_KM
                - H0_km
            )
            / B_km
        )
    )


    reference_1_cell_cm3 = (
        n0_cm3
        * np.exp(
            -(
                PROFILE_CELL_ALTITUDE_KM
                - H0_km
            )
            / B_reference_1_km
        )
    )


    reference_2_cell_cm3 = (
        n0_cm3
        * np.exp(
            -(
                PROFILE_CELL_ALTITUDE_KM
                - H0_km
            )
            / B_reference_2_km
        )
    )


    altitude_edges_km = np.arange(
        PROFILE_H_MIN_KM,
        PROFILE_H_MAX_KM
        + PROFILE_H_BIN_KM,
        PROFILE_H_BIN_KM,
        dtype=np.float64,
    )


    rows = []


    for bin_index in range(
        altitude_edges_km.size
        - 1
    ):

        altitude_lower_km = (
            altitude_edges_km[
                bin_index
            ]
        )

        altitude_upper_km = (
            altitude_edges_km[
                bin_index
                + 1
            ]
        )


        # Include the upper endpoint only in the final bin.
        if (
            bin_index
            == altitude_edges_km.size
            - 2
        ):

            altitude_mask = (
                (
                    PROFILE_CELL_ALTITUDE_KM
                    >= altitude_lower_km
                )
                & (
                    PROFILE_CELL_ALTITUDE_KM
                    <= altitude_upper_km
                )
            )

        else:

            altitude_mask = (
                (
                    PROFILE_CELL_ALTITUDE_KM
                    >= altitude_lower_km
                )
                & (
                    PROFILE_CELL_ALTITUDE_KM
                    < altitude_upper_km
                )
            )


        bin_mask = (
            region_mask
            & altitude_mask
        )


        number_of_cells = np.count_nonzero(
            bin_mask
        )

        if number_of_cells == 0:
            continue


        weights = PROFILE_CELL_VOLUME[
            bin_mask
        ]


        xyz_mean_RM = volume_weighted_mean(
            PROFILE_CELL_XYZ_RM[
                bin_mask
            ],
            weights,
        )


        altitude_mean_km = volume_weighted_mean(
            PROFILE_CELL_ALTITUDE_KM[
                bin_mask
            ],
            weights,
        )


        simulation_mean_cm3 = volume_weighted_mean(
            PROFILE_CELL_NA_CM3[
                bin_mask
            ],
            weights,
        )


        reference_mean_cm3 = volume_weighted_mean(
            reference_cell_cm3[
                bin_mask
            ],
            weights,
        )


        reference_1_mean_cm3 = volume_weighted_mean(
            reference_1_cell_cm3[
                bin_mask
            ],
            weights,
        )


        reference_2_mean_cm3 = volume_weighted_mean(
            reference_2_cell_cm3[
                bin_mask
            ],
            weights,
        )


        rows.append(
            [
                xyz_mean_RM[0],
                xyz_mean_RM[1],
                xyz_mean_RM[2],
                altitude_mean_km,
                simulation_mean_cm3,
                reference_mean_cm3,
                reference_1_mean_cm3,
                reference_2_mean_cm3,
            ]
        )


    if not rows:

        raise RuntimeError(
            f"{location_name}: no valid Fluid Cells "
            "were found in the selected local-time "
            "and altitude ranges"
        )


    return np.asarray(
        rows,
        dtype=np.float64,
    )


# ============================================================
# 17.5 Write one combined ASCII table
# ============================================================

def write_Na_local_time_ascii_table(
    output_path,
    table,
    location_name,
):
    """
    Write simulation and reference profiles into one file.
    """

    model = NA_REFERENCE_MODELS[
        location_name
    ]

    center_hour = float(
        MSO_LOCAL_TIME_CENTERS[
            location_name
        ]
    )

    n0_cm3 = float(
        model[
            "n0_cm3"
        ]
    )

    H0_km = float(
        model[
            "H0_km"
        ]
    )

    B_km = float(
        model[
            "B_km"
        ]
    )

    B_error_km = float(
        model[
            "B_error_km"
        ]
    )

    B_reference_1_km = (
        B_km
        - B_error_km
    )

    B_reference_2_km = (
        B_km
        + B_error_km
    )


    header = (
        f"Na+ altitude profile at {location_name}\n"
        "\n"
        "MSO local-time selection:\n"
        f"center_hour = {center_hour:.6f}\n"
        f"half_width_hour = "
        f"{LT_WINDOW_HALF_WIDTH_HOUR:.6f}\n"
        "\n"
        "Altitude-bin settings:\n"
        f"H_min_km = {PROFILE_H_MIN_KM:.6f}\n"
        f"H_max_km = {PROFILE_H_MAX_KM:.6f}\n"
        f"H_bin_km = {PROFILE_H_BIN_KM:.6f}\n"
        "\n"
        "Reference expression:\n"
        "n_ref(H) = n0 * exp(-(H - H0) / B)\n"
        "\n"
        f"n0_cm-3 = {n0_cm3:.12e}\n"
        f"H0_km = {H0_km:.12e}\n"
        f"B_km = {B_km:.12e}\n"
        f"B_error_km = {B_error_km:.12e}\n"
        f"B_reference_1_km = "
        f"{B_reference_1_km:.12e}\n"
        f"B_reference_2_km = "
        f"{B_reference_2_km:.12e}\n"
        "\n"
        "Columns:\n"
        "1  x_RM\n"
        "2  y_RM\n"
        "3  z_RM\n"
        "4  H_km\n"
        "5  nNa_sim_cm-3\n"
        "6  nNa_reference_cm-3\n"
        "7  nNa_reference_1_cm-3"
        "  [B = B_km - B_error_km]\n"
        "8  nNa_reference_2_cm-3"
        "  [B = B_km + B_error_km]"
    )


    np.savetxt(
        output_path,
        table,
        fmt="%.12e",
        delimiter=" ",
        header=header,
        comments="# ",
    )


    print(
        "Written:",
        output_path,
    )

    print(
        f"  region: {location_name}"
    )

    print(
        f"  rows: {table.shape[0]}"
    )

    print(
        f"  B: {B_km:.1f} "
        f"+/- {B_error_km:.1f} km"
    )


# ============================================================
# 17.6 Export dawn, subsolar and dusk data
# ============================================================

NA_ALTITUDE_TABLES = {}

if RUN_NA_ALTITUDE_PROFILES:
    print()
    print("Exporting MSO local-time Na+ altitude profiles")
    print("Mercury radius [km]:", MERCURY_RADIUS_KM)
    for location_name in ("dawn", "subsolar", "dusk"):
        combined_table = build_Na_local_time_altitude_table(location_name)
        NA_ALTITUDE_TABLES[location_name] = combined_table
        write_Na_local_time_ascii_table(
            PROFILE_OUTPUT_DIR / f"{location_name}.dat",
            combined_table,
            location_name,
        )
    print()
    print("Finished exporting Na+ altitude profiles.")
else:
    print("Skipped Na+ altitude profiles (RUN_NA_ALTITUDE_PROFILES = False).")



'''


def export_na_altitude_profiles(data: dict) -> dict:
    """Write dawn, subsolar and dusk profiles, returning their tables."""
    scope = dict(data)
    scope["RUN_NA_ALTITUDE_PROFILES"] = True
    exec(NA_ALTITUDE_IMPLEMENTATION.replace("\n+", "\n"), scope)
    return scope["NA_ALTITUDE_TABLES"]
