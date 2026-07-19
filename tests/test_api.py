"""Tests for the public API-first data, selection, export, and flux layers."""

from types import SimpleNamespace

import numpy as np
import pytest

from mpcns_post import (
    DerivedFieldRegistry,
    FieldCollection,
    MPCNSCase,
    UnitConverter,
    export_fields_tecplot,
    integrate_surface_flux,
)
from mpcns_post.types import (
    CSRConnectivity,
    GlobalFields,
    GlobalGeometry,
    GlobalTopology,
    LocalEntityMap,
)


@pytest.fixture
def api_case():
    cell_ids = np.array([10, 11, 12, 13], dtype=np.int64)
    cell_xyz = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ]
    )
    face_ids = np.array([100, 101], dtype=np.int64)
    face_xyz = np.array([[-0.5, 0.0, 0.0], [1.5, 0.0, 0.0]])
    geometry = GlobalGeometry(
        node_gid=np.arange(4, dtype=np.int64),
        node_xyz=cell_xyz.copy(),
        edge_gid=np.array([20], dtype=np.int64),
        edge_node_ids=np.array([[0, 1]], dtype=np.int64),
        edge_center_xyz=np.array([[0.5, 0.0, 0.0]]),
        edge_dr=np.array([[1.0, 0.0, 0.0]]),
        edge_length=np.ones(1),
        edge_flags=np.zeros(1, dtype=np.uint32),
        face_gid=face_ids,
        face_center_xyz=face_xyz,
        face_area_vector=np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        face_area=np.ones(2),
        face_flags=np.zeros(2, dtype=np.uint32),
        cell_gid=cell_ids,
        cell_center_xyz=cell_xyz,
        cell_volume=np.ones(4),
        cell_flags=np.ones(4, dtype=np.uint32),
    )
    cell_map = LocalEntityMap(
        0,
        "cell",
        (2, 2, 1),
        cell_ids,
        np.ones(4, dtype=np.int32),
        np.ones(4, dtype=bool),
    )
    empty = CSRConnectivity(
        np.array([0], dtype=np.int64),
        np.empty(0, dtype=np.int64),
        row_global_ids=np.empty(0, dtype=np.int64),
    )
    face_to_cell = CSRConnectivity(
        np.array([0, 1, 2], dtype=np.int64),
        np.array([10, 11], dtype=np.int64),
        row_global_ids=face_ids,
    )
    cell_to_face = CSRConnectivity(
        np.array([0, 1, 2], dtype=np.int64),
        face_ids,
        np.array([-1, 1], dtype=np.int32),
        row_global_ids=np.array([10, 11], dtype=np.int64),
    )

    gamma = 5 / 3
    h_density = np.ones(4)
    h_velocity = np.broadcast_to([1.0, 0.0, 0.0], (4, 3)).copy()
    h_pressure = np.full(4, 2.0)
    h_energy = h_pressure / (gamma - 1) + 0.5 * h_density
    u_h = np.column_stack((h_density, h_density[:, None] * h_velocity, h_energy))

    na_density = np.full(4, 0.5)
    na_velocity = np.broadcast_to([2.0, 0.0, 0.0], (4, 3)).copy()
    na_pressure = np.ones(4)
    na_energy = na_pressure / (gamma - 1) + 0.5 * na_density * 4.0
    u_na = np.column_stack(
        (na_density, na_density[:, None] * na_velocity, na_energy)
    )

    case = object.__new__(MPCNSCase)
    case.directory = None
    case.manifest = SimpleNamespace(
        normalization={
            "density_ref": 2.0,
            "velocity_ref": 3.0,
            "pressure_ref": 5.0,
            "length_ref": 2.0,
        },
        physical_constants={
            "gamma": gamma,
            "particle_mass_H": 1.0,
            "particle_mass_Na": 2.0,
        },
        existing_dynamic_data={
            "fields": [
                {"name": "U_H", "location": "cell"},
                {"name": "U_Na", "location": "cell"},
            ]
        },
        fields=(),
        cell_flag_bits={"fluid": 1, "solid": 2},
    )
    case.geometry = geometry
    case.topology = GlobalTopology(
        [cell_map],
        empty,
        empty,
        face_to_cell,
        cell_to_face,
    )
    case.rank_topologies = [SimpleNamespace(rank=0, local_maps=[cell_map])]
    case.rank_constant_fields = []
    case.unit_converter = UnitConverter(case.manifest.normalization)
    case._latest = [SimpleNamespace(time=0.25)]
    case._dynamic = GlobalFields(
        {"U_H": u_h, "U_Na": u_na},
        {"U_H": cell_ids, "U_Na": cell_ids},
        {
            "U_H": np.ones(4, dtype=bool),
            "U_Na": np.ones(4, dtype=bool),
        },
    )
    case._constant_cache = {}
    case._species_cache = {}
    case.derived = DerivedFieldRegistry(case)
    case.fields = FieldCollection(case)
    case._register_builtin_fields()
    return case


def test_entity_block_and_selection_api(api_case):
    assert api_case.cells.global_ids.tolist() == [10, 11, 12, 13]
    block = next(api_case.iter_blocks(location="cell"))
    np.testing.assert_array_equal(block.indices, [0, 1, 2, 3])
    np.testing.assert_array_equal(
        block.reshape(np.arange(4))[:, :, 0],
        [[0, 2], [1, 3]],
    )

    plane = api_case.select_plane(axis="y", value=0.0, tolerance=0.0)
    np.testing.assert_array_equal(plane.indices, [0, 1])
    np.testing.assert_array_equal(plane.global_ids, [10, 11])
    np.testing.assert_array_equal(plane.mask, [True, True, False, False])

    box = api_case.select_box(x=(0.5, 1.5), y=(-0.1, 1.1))
    np.testing.assert_array_equal(box.global_ids, [11, 13])
    sphere = api_case.select_sphere(center=(0, 0, 0), radius=0.1)
    np.testing.assert_array_equal(sphere.global_ids, [10])


def test_species_and_registered_derived_fields(api_case):
    np.testing.assert_allclose(api_case.H.mass_density(), 2.0)
    np.testing.assert_allclose(
        api_case.Na.velocity_in("m/s"),
        np.broadcast_to([6.0, 0.0, 0.0], (4, 3)),
    )
    np.testing.assert_allclose(api_case.fields["Na_plus_fraction"], 0.2)
    np.testing.assert_allclose(
        api_case.fields["H_Na_drift_velocity"],
        np.broadcast_to([-1.0, 0.0, 0.0], (4, 3)),
    )
    np.testing.assert_allclose(api_case.fields["total_ion_pressure"], 3.0)

    @api_case.register_derived_field("ion_speed_sum", units="normalized")
    def ion_speed_sum(case):
        return np.linalg.norm(case.H.velocity, axis=1) + np.linalg.norm(
            case.Na.velocity,
            axis=1,
        )

    np.testing.assert_allclose(api_case.get_field("ion_speed_sum"), 3.0)
    assert api_case.field_location("ion_speed_sum") == "cell"


def test_generic_tecplot_full_and_plane_export(api_case, tmp_path):
    full = export_fields_tecplot(
        api_case,
        {
            "Na_fraction": api_case.fields["Na_plus_fraction"],
            "drift": api_case.fields["H_Na_drift_velocity"],
        },
        tmp_path / "full.plt",
    )
    assert full.variables == (
        "X",
        "Y",
        "Z",
        "Na_fraction",
        "drift_x",
        "drift_y",
        "drift_z",
    )
    assert full.zones[0][1] == (2, 2, 1)

    plane = api_case.select_plane(axis="y", value=0.0)
    section = export_fields_tecplot(
        api_case,
        {"pressure": api_case.fields["total_ion_pressure"]},
        tmp_path / "plane.plt",
        selection=plane,
    )
    assert section.zones[0][1] == (2, 1, 1)


def test_owner_only_outward_surface_and_flux(api_case):
    surface = api_case.select_boundary_faces()
    np.testing.assert_array_equal(surface.global_ids, [100, 101])
    np.testing.assert_allclose(
        surface.area_vectors,
        [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
    )

    density = np.ones(4)
    velocity = np.broadcast_to([1.0, 0.0, 0.0], (4, 3))
    normalized = integrate_surface_flux(
        api_case,
        surface,
        density,
        velocity,
        dimensional=False,
    )
    np.testing.assert_allclose(normalized.contributions, [-1.0, 1.0])
    assert normalized.total == pytest.approx(0.0)

    right_plane = api_case.select_plane(
        axis="x",
        value=1.5,
        location="face",
    )
    right_surface = api_case.select_boundary_faces(selection=right_plane)
    dimensional = integrate_surface_flux(
        api_case,
        right_surface,
        density,
        velocity,
    )
    # rho_ref * u_ref * length_ref^2 = 2 * 3 * 4 kg/s.
    assert dimensional.total == pytest.approx(24.0)
    assert dimensional.units == "kg/s"

    particle = api_case.species_flux("Na", right_surface, kind="particle")
    # rho_Na=0.5, rho_ref=2, m_Na=2, u_Na=2, u_ref=3, area=4.
    assert particle.total == pytest.approx(12.0)
    assert particle.units == "s^-1"
