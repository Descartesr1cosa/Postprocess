"""
Read MPCNS binary output, compute H+/Na+ and electromagnetic variables,
and project shared Cell values to Nodes.

Important
---------
- All physical calculations are first performed on global Cell arrays.
- The stored cell_scalar_to_node reconstruction is then used component-wise.
- Only Fluid blocks are written.
- The current density is the post-processing curvilinear curl of induced B,
  not the solver's edge/mimetic current.
- The ambipolar electric field needs an electron-pressure closure. Edit
  ELECTRON_PRESSURE_MODEL below to match the physical model.
"""

from pathlib import Path

import numpy as np

from mpcns_post.assemble import GlobalIDIndex
from mpcns_post.errors import ValidationError
from mpcns_post.tecplot import (
    project_cell_values_to_nodes,
)
from post_case_io import load_case


# ============================================================
# 1. User settings
# ============================================================

DIR = Path(globals().get(
    "CASE_DIR",
    r"E:\\2_ClassFiles\\x2025\\Autumn\\Mercury\\python\\Postprocess\\out28",
))
OUTPUT_DIR = DIR / "tecplot_output"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# False:
#     普通生产 restart，不要求保存 J_xi/J_eta/J_zeta
#
# True:
#     debug restart 中必须存在 J_xi/J_eta/J_zeta，
#     并检查 Python DEC 重建和求解器输出是否一致
VALIDATE_DEC_WITH_DEBUG_JEDGE = globals().get(
    "VALIDATE_DEC_WITH_DEBUG_JEDGE",
    False,
)

# Electron-pressure closure used only for the ambipolar field:
#
# "ion_pressure_sum":
#     p_e = p_H + p_Na
#
# "constant_temperature":
#     p_e = n_e k_B T_e
#
# The binary output does not currently contain an independent electron pressure.
ELECTRON_PRESSURE_MODEL = "ion_pressure_sum"

# Used only when ELECTRON_PRESSURE_MODEL == "constant_temperature".
ELECTRON_TEMPERATURE_K = 1.0e5


# Physical constants.
ELEMENTARY_CHARGE_C = 1.602176634e-19
BOLTZMANN_J_K = 1.380649e-23


# ============================================================
# 2. Small helper functions
# ============================================================

def reconstruct_additive_B_cell(case):
    """
    Reconstruct the saved additive Face magnetic field onto global Cells.

    Returns
    -------
    B_add_cell_nd : ndarray, shape (Ncell, 3)
        Nondimensional Cartesian additive magnetic field.
    """

    dynamic = case.dynamic_fields

    face_gids = dynamic.global_ids["B_xi"]
    face_index = GlobalIDIndex.build(face_gids)

    B_add_face = np.zeros(face_gids.size, dtype=np.float64)

    for field_name in ("Badd_xi", "Badd_eta", "Badd_zeta"):

        chunks = [
            chunk
            for chunk in case.rank_constant_fields
            if field_name in chunk.fields
        ]

        if not chunks:
            raise KeyError(f"Missing constant field {field_name}")

        gids = np.concatenate(
            [chunk.global_ids[field_name] for chunk in chunks]
        )

        values = np.concatenate(
            [np.asarray(chunk.fields[field_name]).reshape(-1) for chunk in chunks]
        )

        B_add_face[face_index.lookup(gids)] = values

    operator = case.reconstruction.B_face_to_cell

    B_add_local = operator.apply(
        B_add_face,
        face_index,
    )

    B_add_cell = np.full(
        (case.cells.size, 3),
        np.nan,
        dtype=np.float64,
    )

    cell_index = GlobalIDIndex.build(
        case.cells.global_ids
    )

    B_add_cell[
        cell_index.lookup(operator.output_global_ids)
    ] = B_add_local

    if not np.all(np.isfinite(B_add_cell)):
        raise ValidationError(
            "Additive magnetic-field reconstruction left unfilled Cells"
        )

    return B_add_cell

def curvilinear_gradient(xyz, scalar):
    """
    Cartesian gradient of a scalar on one structured curvilinear block.

    Parameters
    ----------
    xyz : ndarray, shape (ni, nj, nk, 3)
        Physical Cartesian coordinates.
    scalar : ndarray, shape (ni, nj, nk)
        Scalar field.

    Returns
    -------
    gradient : ndarray, shape (ni, nj, nk, 3)
    """

    xyz = np.asarray(xyz, dtype=np.float64)
    scalar = np.asarray(scalar, dtype=np.float64)

    if (
        xyz.ndim != 4
        or xyz.shape[-1] != 3
        or scalar.shape != xyz.shape[:3]
        or min(xyz.shape[:3]) < 2
    ):
        raise ValidationError(
            "Curvilinear gradient requires xyz (ni,nj,nk,3) "
            "and scalar (ni,nj,nk), with every axis >= 2"
        )

    edge_order = 2 if min(xyz.shape[:3]) >= 3 else 1

    # M[..., computational_axis, Cartesian_component]
    #   = d x_component / d computational_axis
    metric = np.stack(
        [
            np.stack(
                np.gradient(
                    xyz[..., component],
                    axis=(0, 1, 2),
                    edge_order=edge_order,
                ),
                axis=-1,
            )
            for component in range(3)
        ],
        axis=-1,
    )

    metric = np.swapaxes(
        metric,
        -1,
        -2,
    )

    d_scalar = np.stack(
        np.gradient(
            scalar,
            axis=(0, 1, 2),
            edge_order=edge_order,
        ),
        axis=-1,
    )

    gradient = np.empty(
        scalar.shape + (3,),
        dtype=np.float64,
    )

    determinant = np.linalg.det(metric)
    scale = np.linalg.norm(metric, axis=(-2, -1))

    good = (
        np.abs(determinant)
        > 1.0e-12 * np.maximum(scale**3, 1.0e-300)
    )

    gradient[good] = np.linalg.solve(
        metric[good],
        d_scalar[good, ..., None],
    )[..., 0]

    if np.any(~good):
        gradient[~good] = np.einsum(
            "...ij,...j->...i",
            np.linalg.pinv(metric[~good]),
            d_scalar[~good],
        )

    if not np.all(np.isfinite(gradient)):
        raise ValidationError(
            "Curvilinear gradient produced non-finite values"
        )

    return gradient


def compute_gradient_on_fluid_cells(case, scalar_cell):
    """
    Compute a physical Cartesian gradient only on Fluid Cell blocks.

    The coordinates and scalar must already use physical SI units.
    """

    gradient = np.full(
        (case.cells.size, 3),
        np.nan,
        dtype=np.float64,
    )

    xyz_m = case.unit_converter.convert(
        case.cells.coordinates,
        quantity="length",
        unit="m",
    )

    for block in case.iter_blocks(location="cell"):

        if block.physics != "Fluid":
            continue

        xyz_block = block.reshape(
            xyz_m
        )

        scalar_block = block.reshape(
            scalar_cell
        )

        gradient_block = curvilinear_gradient(
            xyz_block,
            scalar_block,
        )

        gradient[block.indices] = gradient_block.reshape(
            (-1, 3),
            order="F",
        )

    return gradient

# ============================================================
# 3. Read static geometry/topology and latest restart
# ============================================================

loaded = load_case(DIR)
case = loaded.case
units = case.unit_converter
fluid_mask = loaded.fluid_mask


# ============================================================
# 4. H+ primitive and dimensional variables on Cells
# ============================================================

H = case.H

H_density_nd = H.density
H_velocity_nd = H.velocity
H_pressure_nd = H.pressure

H_mass_density_kg_m3 = H.mass_density(
    "kg/m^3"
)

H_number_density_m3 = H.number_density(
    "m^-3"
)

H_number_density_cm3 = H.number_density(
    "cm^-3"
)

H_velocity_m_s = H.velocity_in(
    "m/s"
)

H_velocity_km_s = H.velocity_in(
    "km/s"
)

H_speed_km_s = np.linalg.norm(
    H_velocity_km_s,
    axis=1,
)

H_pressure_Pa = H.pressure_in(
    "Pa"
)

H_pressure_nPa = H.pressure_in(
    "nPa"
)

H_temperature_K = H.temperature_kelvin


# ============================================================
# 5. Na+ primitive and dimensional variables on Cells
# ============================================================

Na = case.Na

Na_density_nd = Na.density
Na_velocity_nd = Na.velocity
Na_pressure_nd = Na.pressure

Na_mass_density_kg_m3 = Na.mass_density(
    "kg/m^3"
)

Na_number_density_m3 = Na.number_density(
    "m^-3"
)

Na_number_density_cm3 = Na.number_density(
    "cm^-3"
)

Na_velocity_m_s = Na.velocity_in(
    "m/s"
)

Na_velocity_km_s = Na.velocity_in(
    "km/s"
)

Na_speed_km_s = np.linalg.norm(
    Na_velocity_km_s,
    axis=1,
)

Na_pressure_Pa = Na.pressure_in(
    "Pa"
)

Na_pressure_nPa = Na.pressure_in(
    "nPa"
)

Na_temperature_K = Na.temperature_kelvin


# ============================================================
# 6. Electron density and charge-weighted ion velocity
# ============================================================

# Both H+ and Na+ are assumed singly ionized:
#     n_e = n_H+ + n_Na+
electron_number_density_m3 = (
    H_number_density_m3
    + Na_number_density_m3
)

electron_number_density_cm3 = (
    electron_number_density_m3
    * 1.0e-6
)

if np.any(
    electron_number_density_m3[fluid_mask] <= 0.0
):
    raise ValidationError(
        "Non-positive electron number density in Fluid cells"
    )


# Charge-weighted ion velocity:
#     u_plus = (n_H u_H + n_Na u_Na) / n_e
ion_charge_velocity_m_s = np.full(
    (case.cells.size, 3),
    np.nan,
    dtype=np.float64,
)

ion_charge_velocity_m_s[fluid_mask] = (
    H_number_density_m3[fluid_mask, None]
    * H_velocity_m_s[fluid_mask]
    +
    Na_number_density_m3[fluid_mask, None]
    * Na_velocity_m_s[fluid_mask]
) / electron_number_density_m3[fluid_mask, None]

ion_charge_velocity_km_s = (
    ion_charge_velocity_m_s
    * 1.0e-3
)


# ============================================================
# 7. Magnetic field on Cells
# ============================================================

B_induced_cell_nd = case.reconstruct_B_cell(
    case.dynamic_fields
)

B_additive_cell_nd = reconstruct_additive_B_cell(
    case
)

B_total_cell_nd = (
    B_induced_cell_nd
    + B_additive_cell_nd
)


B_induced_T = units.convert(
    B_induced_cell_nd,
    quantity="magnetic_field",
    unit="T",
)

B_additive_T = units.convert(
    B_additive_cell_nd,
    quantity="magnetic_field",
    unit="T",
)

B_total_T = units.convert(
    B_total_cell_nd,
    quantity="magnetic_field",
    unit="T",
)


B_induced_nT = units.convert(
    B_induced_cell_nd,
    quantity="magnetic_field",
    unit="nT",
)

B_additive_nT = units.convert(
    B_additive_cell_nd,
    quantity="magnetic_field",
    unit="nT",
)

B_total_nT = units.convert(
    B_total_cell_nd,
    quantity="magnetic_field",
    unit="nT",
)

B_total_magnitude_nT = np.linalg.norm(
    B_total_nT,
    axis=1,
)


# ============================================================
# 8. Solver-equivalent DEC current
# ============================================================

# 一次完成：
#
# B_face 2-form
#     -> DEC J_edge 1-form
#     -> Cartesian J_cell
#
dec_current = case.reconstruct_current_dec(
    validate_debug=VALIDATE_DEC_WITH_DEBUG_JEDGE,
)


# ------------------------------------------------------------
# Edge DEC current
# ------------------------------------------------------------

# 全局 quotient Edge ID
J_edge_global_ids = (
    dec_current.edge_global_ids
)

# 无量纲 Edge 1-form:
#
#     J_edge = J dot dr
#
# 注意：
# 它不是 Cartesian Jx/Jy/Jz
J_edge_1form_nd = (
    dec_current.edge_1form
)


# ------------------------------------------------------------
# Cartesian Cell current density
# ------------------------------------------------------------

# 无量纲 Cartesian Cell current
#
# shape = (Ncell, 3)
J_induced_cell_nd = (
    dec_current.cell_vector
)


# 转换到 A/m^2
J_induced_A_m2 = units.convert(
    J_induced_cell_nd,
    quantity="current_density",
    unit="A/m^2",
)


# 转换到 nA/m^2
J_induced_nA_m2 = units.convert(
    J_induced_cell_nd,
    quantity="current_density",
    unit="nA/m^2",
)


# 电流强度
J_induced_magnitude_nA_m2 = np.linalg.norm(
    J_induced_nA_m2,
    axis=1,
)


print()
print("DEC current reconstruction:")
print(
    "J_edge shape:",
    J_edge_1form_nd.shape,
)

print(
    "J_cell shape:",
    J_induced_cell_nd.shape,
)

print(
    "|J_cell| nA/m^2:",
    np.nanmin(J_induced_magnitude_nA_m2),
    np.nanmax(J_induced_magnitude_nA_m2),
)

print(
    "debug J_edge max error:",
    dec_current.debug_edge_max_abs_error,
)

# ============================================================
# 9. Motional and Hall electric fields on Cells
# ============================================================

E_motional_V_m = np.full(
    (case.cells.size, 3),
    np.nan,
    dtype=np.float64,
)

E_Hall_V_m = np.full_like(
    E_motional_V_m,
    np.nan,
)


# Motional:
#     E_motional = -u_plus x B_total
E_motional_V_m[fluid_mask] = -np.cross(
    ion_charge_velocity_m_s[fluid_mask],
    B_total_T[fluid_mask],
)


# Hall:
#     E_Hall = (J x B_total) / (e n_e)
E_Hall_V_m[fluid_mask] = np.cross(
    J_induced_A_m2[fluid_mask],
    B_total_T[fluid_mask],
) / (
    ELEMENTARY_CHARGE_C
    * electron_number_density_m3[fluid_mask, None]
)


# ============================================================
# 10. Electron pressure and ambipolar electric field on Cells
# ============================================================

electron_pressure_Pa = np.full(
    case.cells.size,
    np.nan,
    dtype=np.float64,
)


if ELECTRON_PRESSURE_MODEL == "ion_pressure_sum":

    # Explicit modeling assumption:
    #     p_e = p_H + p_Na
    electron_pressure_Pa[fluid_mask] = (
        H_pressure_Pa[fluid_mask]
        + Na_pressure_Pa[fluid_mask]
    )

elif ELECTRON_PRESSURE_MODEL == "constant_temperature":

    # Isothermal electron closure:
    #     p_e = n_e k_B T_e
    electron_pressure_Pa[fluid_mask] = (
        electron_number_density_m3[fluid_mask]
        * BOLTZMANN_J_K
        * ELECTRON_TEMPERATURE_K
    )

else:
    raise ValueError(
        "ELECTRON_PRESSURE_MODEL must be "
        "'ion_pressure_sum' or 'constant_temperature'"
    )


gradient_electron_pressure_Pa_m = (
    compute_gradient_on_fluid_cells(
        case,
        electron_pressure_Pa,
    )
)


E_ambipolar_V_m = np.full_like(
    E_motional_V_m,
    np.nan,
)


# Ambipolar:
#     E_ambipolar = -grad(p_e) / (e n_e)
E_ambipolar_V_m[fluid_mask] = (
    -gradient_electron_pressure_Pa_m[fluid_mask]
    / (
        ELEMENTARY_CHARGE_C
        * electron_number_density_m3[fluid_mask, None]
    )
)


E_generalized_V_m = (
    E_motional_V_m
    + E_Hall_V_m
    + E_ambipolar_V_m
)


E_motional_mV_m = (
    E_motional_V_m
    * 1.0e3
)

E_Hall_mV_m = (
    E_Hall_V_m
    * 1.0e3
)

E_ambipolar_mV_m = (
    E_ambipolar_V_m
    * 1.0e3
)

E_generalized_mV_m = (
    E_generalized_V_m
    * 1.0e3
)


# ============================================================
# 11. Expose all Cell arrays for additional post-processing
# ============================================================

CELL_ARRAYS = {
    # Geometry and masks
    "cell_global_ids": case.cells.global_ids,
    "cell_coordinates_RM": case.cells.coordinates,
    "fluid_mask": fluid_mask,

    # H+
    "H_density_nd": H_density_nd,
    "H_velocity_nd": H_velocity_nd,
    "H_pressure_nd": H_pressure_nd,
    "H_mass_density_kg_m3": H_mass_density_kg_m3,
    "H_number_density_cm3": H_number_density_cm3,
    "H_velocity_km_s": H_velocity_km_s,
    "H_pressure_nPa": H_pressure_nPa,
    "H_temperature_K": H_temperature_K,

    # Na+
    "Na_density_nd": Na_density_nd,
    "Na_velocity_nd": Na_velocity_nd,
    "Na_pressure_nd": Na_pressure_nd,
    "Na_mass_density_kg_m3": Na_mass_density_kg_m3,
    "Na_number_density_cm3": Na_number_density_cm3,
    "Na_velocity_km_s": Na_velocity_km_s,
    "Na_pressure_nPa": Na_pressure_nPa,
    "Na_temperature_K": Na_temperature_K,

    # Electron / charge-weighted ion quantities
    "electron_number_density_cm3": electron_number_density_cm3,
    "electron_pressure_Pa": electron_pressure_Pa,
    "ion_charge_velocity_km_s": ion_charge_velocity_km_s,

    # Electromagnetic variables
    "B_induced_nT": B_induced_nT,
    "B_additive_nT": B_additive_nT,
    "B_total_nT": B_total_nT,
    "J_induced_cell_nd": J_induced_cell_nd,
    "J_induced_A_m2": J_induced_A_m2,
    "J_induced_nA_m2": J_induced_nA_m2,
    "E_motional_mV_m": E_motional_mV_m,
    "E_Hall_mV_m": E_Hall_mV_m,
    "E_ambipolar_mV_m": E_ambipolar_mV_m,
    "E_generalized_mV_m": E_generalized_mV_m,
}

EDGE_ARRAYS = {
    "edge_global_ids":
        J_edge_global_ids,

    "edge_coordinates_RM":
        case.edges.coordinates,

    "edge_dr_RM":
        case.geometry.edge_dr,

    "edge_length_RM":
        case.edges.measure,

    "J_edge_1form_nd":
        J_edge_1form_nd,
}

# ============================================================
# 12. Additional magnitudes on Cells
# ============================================================

B_induced_magnitude_nT = np.linalg.norm(
    B_induced_nT,
    axis=1,
)

B_additive_magnitude_nT = np.linalg.norm(
    B_additive_nT,
    axis=1,
)

ion_charge_speed_km_s = np.linalg.norm(
    ion_charge_velocity_km_s,
    axis=1,
)

E_motional_magnitude_mV_m = np.linalg.norm(
    E_motional_mV_m,
    axis=1,
)

E_Hall_magnitude_mV_m = np.linalg.norm(
    E_Hall_mV_m,
    axis=1,
)

E_ambipolar_magnitude_mV_m = np.linalg.norm(
    E_ambipolar_mV_m,
    axis=1,
)

E_generalized_magnitude_mV_m = np.linalg.norm(
    E_generalized_mV_m,
    axis=1,
)


# Add the extra Cell arrays to the exposed dictionary.
CELL_ARRAYS.update({
    "B_induced_magnitude_nT":
        B_induced_magnitude_nT,

    "B_additive_magnitude_nT":
        B_additive_magnitude_nT,

    "B_total_magnitude_nT":
        B_total_magnitude_nT,

    "J_induced_magnitude_nA_m2":
        J_induced_magnitude_nA_m2,

    "ion_charge_speed_km_s":
        ion_charge_speed_km_s,

    "E_motional_magnitude_mV_m":
        E_motional_magnitude_mV_m,

    "E_Hall_magnitude_mV_m":
        E_Hall_magnitude_mV_m,

    "E_ambipolar_magnitude_mV_m":
        E_ambipolar_magnitude_mV_m,

    "E_generalized_magnitude_mV_m":
        E_generalized_magnitude_mV_m,
})


# ============================================================
# 13. Project Cell variables to global Nodes
# ============================================================

def to_fluid_nodes(values):
    """
    Project one global Cell array onto global Nodes.

    Solid Cells are excluded through fluid_mask and the remaining
    Cell-to-Node weights are renormalized.
    """

    return project_cell_values_to_nodes(
        case,
        values,
        valid_mask=fluid_mask,
    )

# ============================================================
# Fluid output: H+ and Na+ primitive variables at Nodes
# ============================================================

NODE_FLUID = {
    # H+
    "<greek>r</greek><times><sub>H+</sub> (kg/m<sup>3</sup>)":
        to_fluid_nodes(H_mass_density_kg_m3),

    "<times><i>n</i><sub>H+</sub> (/cm<sup>3</sup>)":
        to_fluid_nodes(H_number_density_cm3),

    "<times><i><b>u</b></i><sub>H+,{c}</sub> (km/s)":
        to_fluid_nodes(H_velocity_km_s),

    "<times>|<i><b>u</b></i>|<sub>H+</sub> (km/s)":
        to_fluid_nodes(H_speed_km_s),

    "<times><i>p</i><sub>H+</sub> (nPa)":
        to_fluid_nodes(H_pressure_nPa),

    "<times><i>T</i><sub>H+</sub> (K)":
        to_fluid_nodes(H_temperature_K),

    # Na+
    "<greek>r</greek><times><sub>Na+</sub> (kg/m<sup>3</sup>)":
        to_fluid_nodes(Na_mass_density_kg_m3),

    "<times><i>n</i><sub>Na+</sub> (/cm<sup>3</sup>)":
        to_fluid_nodes(Na_number_density_cm3),

    "<times><i><b>u</b></i><sub>Na+,{c}</sub> (km/s)":
        to_fluid_nodes(Na_velocity_km_s),

    "<times>|<i><b>u</b></i>|<sub>Na+</sub> (km/s)":
        to_fluid_nodes(Na_speed_km_s),

    "<times><i>p</i><sub>Na+</sub> (nPa)":
        to_fluid_nodes(Na_pressure_nPa),

    "<times><i>T</i><sub>Na+</sub> (K)":
        to_fluid_nodes(Na_temperature_K),
}

# ------------------------------------------------------------
# Output 1:
# Total magnetic field + magnitude
# Current density + magnitude
# Total electric field + magnitude
# ------------------------------------------------------------

NODE_EM_TOTAL = {
    "<times><i><b>B</b></i><sub>{c}</sub> (nT)":
        to_fluid_nodes(B_total_nT),

    "<times>|<i><b>B</b></i>| (nT)":
        to_fluid_nodes(B_total_magnitude_nT),

    "<times><i><b>J</b></i><sub>{c}</sub> (nA/m<sup>2</sup>)":
        to_fluid_nodes(J_induced_nA_m2),

    "<times>|<i><b>J</b></i>| (nA/m<sup>2</sup>)":
        to_fluid_nodes(J_induced_magnitude_nA_m2),

    "<times><i><b>E</b></i><sub>{c}</sub> (mV/m)":
        to_fluid_nodes(E_generalized_mV_m),

    "<times>|<i><b>E</b></i>| (mV/m)":
        to_fluid_nodes(E_generalized_magnitude_mV_m),
}


# ------------------------------------------------------------
# Output 2:
# Induced magnetic field and additive/background magnetic field
# ------------------------------------------------------------

NODE_EM_B_DECOMPOSITION = {
    "<times><i><b>B</b></i><sub>ind,{c}</sub> (nT)":
        to_fluid_nodes(B_induced_nT),

    "<times>|<i><b>B</b></i><sub>ind</sub>| (nT)":
        to_fluid_nodes(B_induced_magnitude_nT),

    "<times><i><b>B</b></i><sub>add,{c}</sub> (nT)":
        to_fluid_nodes(B_additive_nT),

    "<times>|<i><b>B</b></i><sub>add</sub>| (nT)":
        to_fluid_nodes(B_additive_magnitude_nT),
}


# ------------------------------------------------------------
# Output 3:
# Separate electric-field terms
# ------------------------------------------------------------

NODE_EM_E_COMPONENTS = {
    "<times><i><b>E</b></i><sub>motion,{c}</sub> (mV/m)":
        to_fluid_nodes(E_motional_mV_m),

    "<times>|<i>E</i><sub>motion</sub>| (mV/m)":
        to_fluid_nodes(E_motional_magnitude_mV_m),

    "<times><i><b>E</b></i><sub>Hall,{c}</sub> (mV/m)":
        to_fluid_nodes(E_Hall_mV_m),

    "<times>|<i>E</i><sub>Hall</sub>| (mV/m)":
        to_fluid_nodes(E_Hall_magnitude_mV_m),

    "<times><i><b>E</b></i><sub>amb,{c}</sub> (mV/m)":
        to_fluid_nodes(E_ambipolar_mV_m),

    "<times>|<i>E</i><sub>amb</sub>| (mV/m)":
        to_fluid_nodes(E_ambipolar_magnitude_mV_m),
}


# ------------------------------------------------------------
# Output 4:
# Other quantities used by the generalized Ohm-law terms
# ------------------------------------------------------------

NODE_EM_AUXILIARY = {
    "<times><i>n</i><sub>e</sub> (/cm<sup>3</sup>)":
        to_fluid_nodes(electron_number_density_cm3),

    "<times><i>p</i><sub>e</sub> (nPa)":
        to_fluid_nodes(electron_pressure_Pa * 1.0e9),

    "<times><i><b>u</b></i><sub>+,{c}</sub> (km/s)":
        to_fluid_nodes(ion_charge_velocity_km_s),

    "<times>|<i><b>u</b></i><sub>+</sub>| (km/s)":
        to_fluid_nodes(ion_charge_speed_km_s),
}


# All Node arrays remain available for extra post-processing.
NODE_ARRAYS = {
    "node_global_ids":
        case.nodes.global_ids,

    "node_coordinates_RM":
        case.nodes.coordinates,

    # H+
    "H_mass_density_kg_m3":
        to_fluid_nodes(H_mass_density_kg_m3),

    "H_number_density_cm3":
        to_fluid_nodes(H_number_density_cm3),

    "H_velocity_km_s":
        to_fluid_nodes(H_velocity_km_s),

    "H_speed_km_s":
        to_fluid_nodes(H_speed_km_s),

    "H_pressure_nPa":
        to_fluid_nodes(H_pressure_nPa),

    "H_temperature_K":
        to_fluid_nodes(H_temperature_K),

    # Na+
    "Na_mass_density_kg_m3":
        to_fluid_nodes(Na_mass_density_kg_m3),

    "Na_number_density_cm3":
        to_fluid_nodes(Na_number_density_cm3),

    "Na_velocity_km_s":
        to_fluid_nodes(Na_velocity_km_s),

    "Na_speed_km_s":
        to_fluid_nodes(Na_speed_km_s),

    "Na_pressure_nPa":
        to_fluid_nodes(Na_pressure_nPa),

    "Na_temperature_K":
        to_fluid_nodes(Na_temperature_K),

    "B_total_nT":
        to_fluid_nodes(B_total_nT),

    "B_total_magnitude_nT":
        to_fluid_nodes(B_total_magnitude_nT),

    "J_induced_nA_m2":
        to_fluid_nodes(J_induced_nA_m2),

    "J_induced_magnitude_nA_m2":
        to_fluid_nodes(J_induced_magnitude_nA_m2),

    "E_generalized_mV_m":
        to_fluid_nodes(E_generalized_mV_m),

    "E_generalized_magnitude_mV_m":
        to_fluid_nodes(E_generalized_magnitude_mV_m),

    "B_induced_nT":
        to_fluid_nodes(B_induced_nT),

    "B_induced_magnitude_nT":
        to_fluid_nodes(B_induced_magnitude_nT),

    "B_additive_nT":
        to_fluid_nodes(B_additive_nT),

    "B_additive_magnitude_nT":
        to_fluid_nodes(B_additive_magnitude_nT),

    "E_motional_mV_m":
        to_fluid_nodes(E_motional_mV_m),

    "E_motional_magnitude_mV_m":
        to_fluid_nodes(E_motional_magnitude_mV_m),

    "E_Hall_mV_m":
        to_fluid_nodes(E_Hall_mV_m),

    "E_Hall_magnitude_mV_m":
        to_fluid_nodes(E_Hall_magnitude_mV_m),

    "E_ambipolar_mV_m":
        to_fluid_nodes(E_ambipolar_mV_m),

    "E_ambipolar_magnitude_mV_m":
        to_fluid_nodes(E_ambipolar_magnitude_mV_m),

    "electron_number_density_cm3":
        to_fluid_nodes(electron_number_density_cm3),

    "electron_pressure_nPa":
        to_fluid_nodes(electron_pressure_Pa * 1.0e9),

    "ion_charge_velocity_km_s":
        to_fluid_nodes(ion_charge_velocity_km_s),

    "ion_charge_speed_km_s":
        to_fluid_nodes(ion_charge_speed_km_s),
}
