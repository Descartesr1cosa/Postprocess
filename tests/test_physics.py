"""Tests for derived physical quantities and numerical diagnostics."""

import numpy as np

from mpcns_post.derived import (
    conserved_to_primitive,
    curvilinear_curl,
    neutral_sodium_from_photo_rate,
    species_temperature_kelvin,
)


def test_species_energy_excludes_magnetic_energy():
    density = 2.0
    velocity = np.array([1.0, 2.0, 3.0])
    pressure = 5.0
    gamma = 5 / 3
    energy = 0.5 * density * np.dot(velocity, velocity) + pressure / (gamma - 1)
    conserved = np.r_[density, density * velocity, energy][None]

    primitive = conserved_to_primitive(conserved, gamma=gamma)

    np.testing.assert_allclose(primitive.velocity, velocity[None])
    np.testing.assert_allclose(primitive.pressure, pressure)


def test_cartesian_curl():
    axis = np.linspace(-1, 1, 5)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    xyz = np.stack((x, y, z), axis=-1)
    # B=(-y/2, x/2, 0) has curl=(0, 0, 1).
    magnetic_field = np.stack((-0.5 * y, 0.5 * x, np.zeros_like(z)), axis=-1)

    curl = curvilinear_curl(xyz, magnetic_field)

    np.testing.assert_allclose(
        curl,
        np.broadcast_to([0, 0, 1], curl.shape),
        atol=1e-12,
    )


def test_neutral_sodium_from_photo_rate():
    xyz = np.array([[2.0, 0, 0], [-2.0, 0, 0], [-2.0, 2, 0]])
    neutral = np.array([10.0, 20.0, 30.0])
    frequency = np.array([5e-5, 1e-5, 5e-5])

    np.testing.assert_allclose(
        neutral_sodium_from_photo_rate(neutral * frequency, xyz),
        neutral,
    )


def test_species_temperature():
    boltzmann = 1.380649e-23
    particle_mass = 2.0
    temperature = 400.0
    density = np.array([3.0]) * particle_mass
    pressure = np.array([3.0]) * boltzmann * temperature

    np.testing.assert_allclose(
        species_temperature_kelvin(
            density,
            pressure,
            density_ref=1,
            pressure_ref=1,
            particle_mass=particle_mass,
        ),
        temperature,
    )
