"""Owner-only boundary-surface selection and orientation-safe flux integrals."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .access import Selection
from .assemble import GlobalIDIndex
from .errors import ValidationError


@dataclass(frozen=True)
class SurfaceSelection:
    """An owner-only set of exterior faces with outward area vectors."""

    indices: np.ndarray
    coordinates: np.ndarray
    global_ids: np.ndarray
    mask: np.ndarray
    owner_cell_indices: np.ndarray
    area_vectors: np.ndarray
    orientation_sign: np.ndarray

    @property
    def location(self) -> str:
        return "face"

    def __len__(self) -> int:
        return int(self.indices.size)


@dataclass(frozen=True)
class FluxResult:
    """Integrated signed outward flux and its per-face contributions."""

    total: float
    contributions: np.ndarray
    units: str
    surface: SurfaceSelection
    interpolation: str


def _relation_rows(relation) -> dict[int, tuple[np.ndarray, np.ndarray | None]]:
    if relation.row_global_ids is None:
        raise ValidationError("global topology relation has no row global IDs")
    rows = {}
    for row_index, row_gid in enumerate(relation.row_global_ids):
        start, stop = relation.offsets[row_index:row_index + 2]
        signs = None if relation.signs is None else relation.signs[start:stop]
        rows[int(row_gid)] = (relation.indices[start:stop], signs)
    return rows


def select_boundary_faces(
    case,
    *,
    selection: Selection | None = None,
    predicate: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
) -> SurfaceSelection:
    """Select unique exterior faces and orient their area vectors outward.

    ``selection`` may come from ``case.select_plane/box/sphere(location="face")``.
    A predicate receives all face centers and stored directed area vectors.
    Only faces incident to exactly one global Cell are retained.
    """
    faces = case.entity("face")
    if faces.area_vectors is None:
        raise ValidationError("face area vectors are unavailable")
    face_rows = _relation_rows(case.topology.face_to_cell)
    if case.topology.cell_to_face is None:
        raise ValidationError("cell_to_face orientation signs are unavailable")
    cell_rows = _relation_rows(case.topology.cell_to_face)
    cell_index = GlobalIDIndex.build(case.geometry.cell_gid)

    boundary = np.zeros(faces.size, dtype=bool)
    owner_gids = np.full(faces.size, -1, dtype=np.int64)
    incidence_sign = np.ones(faces.size, dtype=np.int32)
    for face_index, face_gid in enumerate(faces.global_ids):
        row = face_rows.get(int(face_gid))
        if row is None or row[0].size != 1:
            continue
        owner_gid = int(row[0][0])
        cell_row = cell_rows.get(owner_gid)
        if cell_row is None or cell_row[1] is None:
            raise ValidationError(f"missing oriented cell_to_face row for Cell {owner_gid}")
        matches = np.flatnonzero(cell_row[0] == face_gid)
        if matches.size != 1:
            raise ValidationError(
                f"Cell {owner_gid} contains Face {int(face_gid)} {matches.size} times"
            )
        boundary[face_index] = True
        owner_gids[face_index] = owner_gid
        incidence_sign[face_index] = int(cell_row[1][matches[0]])

    mask = boundary
    if selection is not None:
        if selection.location != "face" or selection.mask.shape != mask.shape:
            raise ValueError("surface selection must use the case's global face ordering")
        mask = mask & selection.mask
    if predicate is not None:
        accepted = np.asarray(
            predicate(faces.coordinates, faces.area_vectors),
            dtype=bool,
        )
        if accepted.shape != mask.shape:
            raise ValueError("surface predicate must return one boolean per global face")
        mask = mask & accepted

    indices = np.flatnonzero(mask)
    selected_owner_gids = owner_gids[indices]
    owner_indices = cell_index.lookup(selected_owner_gids)
    signs = incidence_sign[indices].copy()
    area_vectors = faces.area_vectors[indices] * signs[:, None]

    # Incidence signs are authoritative. The center-to-center geometric check
    # catches legacy files with reversed sign convention and makes the returned
    # vector explicitly outward for the one-cell exterior face.
    radial = faces.coordinates[indices] - case.geometry.cell_center_xyz[owner_indices]
    inward = np.einsum("ij,ij->i", area_vectors, radial) < 0
    if np.any(inward):
        signs[inward] *= -1
        area_vectors[inward] *= -1

    return SurfaceSelection(
        indices,
        faces.coordinates[indices],
        faces.global_ids[indices],
        mask,
        owner_indices,
        area_vectors,
        signs,
    )


def integrate_surface_flux(
    case,
    surface: SurfaceSelection,
    density: np.ndarray,
    velocity: np.ndarray,
    *,
    kind: str = "mass",
    particle_mass: float | None = None,
    dimensional: bool = True,
    interpolation: str = "owner",
) -> FluxResult:
    """Integrate ``density * velocity · dS`` over an exterior surface.

    Cell-to-face interpolation is deliberately explicit. Exterior owner-only
    faces currently support only ``interpolation="owner"``; this avoids hidden
    ghost/neighbor averaging and MPI alias double counting.
    """
    if interpolation != "owner":
        raise ValueError("boundary flux currently supports only interpolation='owner'")
    if kind not in {"mass", "particle"}:
        raise ValueError("kind must be 'mass' or 'particle'")
    rho = np.asarray(density, dtype=np.float64)
    velocity_array = np.asarray(velocity, dtype=np.float64)
    cell_count = case.entity("cell").size
    if rho.shape != (cell_count,) or velocity_array.shape != (cell_count, 3):
        raise ValidationError(
            f"flux inputs must have shapes ({cell_count},) and ({cell_count}, 3)"
        )

    owner_density = rho[surface.owner_cell_indices]
    owner_velocity = velocity_array[surface.owner_cell_indices]
    area_vectors = surface.area_vectors
    if dimensional:
        normalization = case.manifest.normalization
        try:
            density_scale = float(normalization["density_ref"])
            velocity_scale = float(normalization["velocity_ref"])
            area_scale = float(normalization["length_ref"]) ** 2
        except KeyError as exc:
            raise ValidationError(
                f"dimensional flux requires normalization value {exc.args[0]!r}"
            ) from exc
        owner_density = owner_density * density_scale
        owner_velocity = owner_velocity * velocity_scale
        area_vectors = area_vectors * area_scale
        if kind == "particle":
            if particle_mass is None or particle_mass <= 0:
                raise ValueError("particle flux requires a positive particle_mass")
            owner_density = owner_density / particle_mass
        units = "s^-1" if kind == "particle" else "kg/s"
    else:
        if kind == "particle":
            raise ValueError("particle flux is defined only for dimensional=True")
        units = "normalized mass flux"

    contributions = owner_density * np.einsum(
        "ij,ij->i",
        owner_velocity,
        area_vectors,
    )
    if np.any(~np.isfinite(contributions)):
        raise ValidationError("surface flux produced non-finite contributions")
    return FluxResult(
        float(np.sum(contributions, dtype=np.float64)),
        contributions,
        units,
        surface,
        interpolation,
    )


def integrate_species_flux(
    case,
    surface: SurfaceSelection,
    *,
    species: str = "Na",
    kind: str = "particle",
    dimensional: bool = True,
    interpolation: str = "owner",
) -> FluxResult:
    """Integrate one ion species' mass or particle escape flux."""
    state = case.species(species)
    return integrate_surface_flux(
        case,
        surface,
        state.density,
        state.velocity,
        kind=kind,
        particle_mass=state.particle_mass,
        dimensional=dimensional,
        interpolation=interpolation,
    )
