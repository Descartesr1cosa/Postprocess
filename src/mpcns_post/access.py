"""High-level entity, block, species, and derived-field access APIs."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .errors import ValidationError
from .derived import species_temperature_kelvin

if TYPE_CHECKING:
    from .case import MPCNSCase


BASE_LOCATIONS = ("cell", "node", "face", "edge")


def base_location(location: str) -> str:
    """Normalize staggered locations such as ``face_xi`` to ``face``."""
    normalized = location.lower()
    for base in BASE_LOCATIONS:
        if normalized == base or normalized.startswith(base + "_"):
            return base
    raise ValueError(f"unsupported entity location {location!r}")


@dataclass(frozen=True)
class EntityView:
    """Owner-only global entity arrays in canonical global-ID order."""

    location: str
    global_ids: np.ndarray
    coordinates: np.ndarray
    flags: np.ndarray | None = None
    measure: np.ndarray | None = None
    area_vectors: np.ndarray | None = None
    connectivity: np.ndarray | None = None

    @property
    def size(self) -> int:
        return int(self.global_ids.size)


@dataclass(frozen=True)
class Selection:
    """A spatial subset with both sparse indices and a full-size mask."""

    location: str
    indices: np.ndarray
    coordinates: np.ndarray
    global_ids: np.ndarray
    mask: np.ndarray

    def apply(self, values: np.ndarray) -> np.ndarray:
        """Select rows from a global field in the same entity ordering."""
        array = np.asarray(values)
        if array.ndim == 0 or array.shape[0] != self.mask.size:
            raise ValidationError(
                f"selection expects {self.mask.size} {self.location} values, "
                f"got shape {array.shape}"
            )
        return array[self.indices]

    def __len__(self) -> int:
        return int(self.indices.size)


@dataclass(frozen=True)
class Block:
    """One rank-local structured block mapped into global entity arrays."""

    rank: int
    block_id: int
    location: str
    logical_shape: tuple[int, int, int]
    global_ids: np.ndarray
    indices: np.ndarray
    owner_mask: np.ndarray
    orientation_sign: np.ndarray
    physics: str | None = None

    @property
    def size(self) -> int:
        return int(self.global_ids.size)

    @property
    def owner_indices(self) -> np.ndarray:
        """Canonical global indices owned by this rank/block map."""
        return self.indices[self.owner_mask]

    def reshape(self, global_field: np.ndarray) -> np.ndarray:
        """Gather and reshape a global array using i-fastest block ordering."""
        values = np.asarray(global_field)
        required = int(self.indices.max()) + 1 if self.indices.size else 0
        if values.ndim == 0 or values.shape[0] < required:
            raise ValidationError(
                f"block requires at least {required} global {base_location(self.location)} "
                f"values, got shape {values.shape}"
            )
        trailing = values.shape[1:]
        return values[self.indices].reshape(self.logical_shape + trailing, order="F")


@dataclass(frozen=True)
class SpeciesData:
    """Nondimensional primitive state plus explicit dimensional conversions."""

    case: "MPCNSCase"
    name: str
    conserved: np.ndarray
    density: np.ndarray
    velocity: np.ndarray
    pressure: np.ndarray
    valid_mask: np.ndarray

    @property
    def particle_mass(self) -> float:
        try:
            return float(self.case.manifest.physical_constants[f"particle_mass_{self.name}"])
        except KeyError as exc:
            raise ValidationError(f"particle mass is unavailable for species {self.name}") from exc

    def mass_density(self, unit: str = "kg/m^3") -> np.ndarray:
        return self.case.unit_converter.convert(self.density, "density", unit)

    def number_density(self, unit: str = "m^-3") -> np.ndarray:
        return self.case.unit_converter.number_density(
            self.density,
            particle_mass=self.particle_mass,
            unit=unit,
        )

    def velocity_in(self, unit: str = "m/s") -> np.ndarray:
        return self.case.unit_converter.convert(self.velocity, "velocity", unit)

    def pressure_in(self, unit: str = "Pa") -> np.ndarray:
        return self.case.unit_converter.convert(self.pressure, "pressure", unit)

    @property
    def temperature_kelvin(self) -> np.ndarray:
        return species_temperature_kelvin(
            self.density,
            self.pressure,
            density_ref=self.case.manifest.normalization["density_ref"],
            pressure_ref=self.case.manifest.normalization["pressure_ref"],
            particle_mass=self.particle_mass,
        )


DerivedCalculator = Callable[["MPCNSCase"], np.ndarray]


@dataclass(frozen=True)
class DerivedField:
    """Metadata and calculator for one lazily evaluated field."""

    name: str
    calculator: DerivedCalculator
    location: str = "cell"
    units: str | None = None
    description: str = ""


class DerivedFieldRegistry:
    """Per-case registry for lazily evaluated, cached derived arrays."""

    def __init__(self, case: "MPCNSCase") -> None:
        self.case = case
        self._definitions: dict[str, DerivedField] = {}
        self._cache: dict[str, np.ndarray] = {}

    def register(
        self,
        name: str,
        calculator: DerivedCalculator | None = None,
        *,
        location: str = "cell",
        units: str | None = None,
        description: str = "",
        overwrite: bool = False,
    ):
        """Register a calculator directly or use this method as a decorator."""
        normalized_location = base_location(location)

        def add(function: DerivedCalculator) -> DerivedCalculator:
            if name in self._definitions and not overwrite:
                raise ValueError(f"derived field {name!r} is already registered")
            self._definitions[name] = DerivedField(
                name,
                function,
                normalized_location,
                units,
                description,
            )
            self._cache.pop(name, None)
            return function

        return add if calculator is None else add(calculator)

    def evaluate(self, name: str) -> np.ndarray:
        if name in self._cache:
            return self._cache[name]
        try:
            definition = self._definitions[name]
        except KeyError as exc:
            raise KeyError(f"unknown derived field {name!r}") from exc
        values = np.asarray(definition.calculator(self.case))
        expected = self.case.entity(definition.location).size
        if values.ndim == 0 or values.shape[0] != expected:
            raise ValidationError(
                f"derived field {name!r} at {definition.location} must have "
                f"leading size {expected}, got {values.shape}"
            )
        self._cache[name] = values
        return values

    def clear_cache(self) -> None:
        self._cache.clear()

    def location(self, name: str) -> str:
        return self._definitions[name].location

    @property
    def definitions(self) -> Mapping[str, DerivedField]:
        return dict(self._definitions)

    def __contains__(self, name: object) -> bool:
        return name in self._definitions


class FieldCollection(Mapping[str, np.ndarray]):
    """Read-only mapping that resolves raw, constant, and derived fields."""

    def __init__(self, case: "MPCNSCase") -> None:
        self.case = case

    def __getitem__(self, name: str) -> np.ndarray:
        return self.case.get_field(name)

    def __iter__(self) -> Iterator[str]:
        return iter(self.case.available_fields)

    def __len__(self) -> int:
        return len(self.case.available_fields)
