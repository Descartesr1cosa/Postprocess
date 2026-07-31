"""Cell-centre X/O-point search on one Cartesian simulation slice.

No field interpolation is used: a detected critical point is reported at the
centre of the Cell that contains its locally fitted zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class PlanePoint:
    kind: str
    xyz_RM: np.ndarray


def _plane_axes(normal_axis: int) -> tuple[int, int]:
    return tuple(axis for axis in range(3) if axis != normal_axis)  # type: ignore[return-value]


def _nearest_type_chain(points: list[PlanePoint]) -> list[PlanePoint]:
    """Connect a non-monotonic cloud, preferring alternating X/O neighbours."""
    remaining = list(points)
    ordered = [remaining.pop(int(np.argmin([point.xyz_RM[0] for point in remaining])))]
    while remaining:
        candidates = [
            index for index, point in enumerate(remaining)
            if point.kind != ordered[-1].kind
        ]
        if not candidates:
            candidates = list(range(len(remaining)))
        next_index = min(
            candidates,
            key=lambda index: np.linalg.norm(remaining[index].xyz_RM - ordered[-1].xyz_RM),
        )
        ordered.append(remaining.pop(next_index))
    return ordered


def _order_plane_group(points: list[PlanePoint], tangent: tuple[int, int]) -> list[PlanePoint]:
    """Use a non-overlapping in-plane coordinate, otherwise chain by X/O type."""
    if len(points) < 3:
        return sorted(points, key=lambda point: point.xyz_RM[tangent[0]])
    coordinates = np.asarray([point.xyz_RM for point in points])
    candidates = []
    for axis in tangent:
        order = np.argsort(coordinates[:, axis], kind="stable")
        projected_gap = np.diff(coordinates[order, axis])
        # A valid monotonic coordinate separates each adjacent point by at
        # least a quarter of the median point separation in the slice.
        pairwise = np.linalg.norm(coordinates[:, None] - coordinates[None, :], axis=2)
        spacing = float(np.median(pairwise[np.triu_indices(len(points), k=1)]))
        if np.min(projected_gap) >= 0.25 * spacing:
            path = float(np.sum(np.linalg.norm(np.diff(coordinates[order], axis=0), axis=1)))
            candidates.append((path, axis, order))
    if candidates:
        _, _, order = min(candidates, key=lambda item: item[0])
        return [points[index] for index in order]
    return _nearest_type_chain(points)


def find_plane_xo_points(xyz_RM: np.ndarray, B_total: np.ndarray, fluid_mask: np.ndarray, *,
                         normal_axis: str, value_rm: float, tolerance_rm: float,
                         fit_neighbours: int = 12, root_radius_in_spacings: float = 1.5) -> list[PlanePoint]:
    """Find X/O points in a Cartesian plane slab using Cell-centred B.

    At every local minimum of the in-plane field, a least-squares 2-D
    Jacobian is fitted from nearby slice Cells.  Its zero must lie within a
    local-cell-scale radius; the sign of its determinant gives X (<0) or O
    (>0).  Only the candidate Cell centre is returned.
    """
    axis = {"x": 0, "y": 1, "z": 2}.get(normal_axis.lower())
    if axis is None:
        raise ValueError("normal_axis must be 'x', 'y', or 'z'")
    if tolerance_rm < 0.0:
        raise ValueError("tolerance_rm must be non-negative")
    tangent = _plane_axes(axis)
    xyz = np.asarray(xyz_RM, dtype=float)
    field = np.asarray(B_total, dtype=float)
    fluid = np.asarray(fluid_mask, dtype=bool)
    selected = fluid & np.isfinite(field).all(axis=1) & (np.abs(xyz[:, axis] - value_rm) <= tolerance_rm)
    source_indices = np.flatnonzero(selected)
    if source_indices.size < max(6, fit_neighbours):
        raise ValueError(
            "Too few Fluid Cells in plane slab; increase PLANE_TOLERANCE_RM "
            "or choose a plane represented by the mesh"
        )
    plane_xyz = xyz[source_indices][:, tangent]
    plane_field = field[source_indices][:, tangent]
    neighbours = min(int(fit_neighbours), source_indices.size)
    tree = cKDTree(plane_xyz)
    distances, nearby = tree.query(plane_xyz, k=neighbours)
    if neighbours == 1:
        distances, nearby = distances[:, None], nearby[:, None]
    spacing = float(np.median(distances[:, 1]))
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("Plane Cells have zero/invalid nearest-neighbour spacing")

    chosen: list[PlanePoint] = []
    for centre, donor_ids in enumerate(nearby):
        donor_ids = np.asarray(donor_ids, dtype=int)
        magnitude = np.linalg.norm(plane_field[donor_ids], axis=1)
        # A real zero is locally smaller than the surrounding sampled field.
        if magnitude[0] > np.min(magnitude[1:]):
            continue
        delta_x = plane_xyz[donor_ids] - plane_xyz[centre]
        delta_b = plane_field[donor_ids] - plane_field[centre]
        if np.linalg.matrix_rank(delta_x) < 2:
            continue
        # delta_B = J delta_x; use a least-squares local 2-D Jacobian.
        jacobian = np.linalg.lstsq(delta_x, delta_b, rcond=None)[0].T
        determinant = float(np.linalg.det(jacobian))
        if not np.isfinite(determinant) or abs(determinant) <= 1.0e-14:
            continue
        root_offset = np.linalg.solve(jacobian, -plane_field[centre])
        if np.linalg.norm(root_offset) > root_radius_in_spacings * spacing:
            continue
        point = PlanePoint("X" if determinant < 0.0 else "O", xyz[source_indices[centre]].copy())
        # Adjacent Cells can identify one zero.  Keep the closest-to-zero Cell.
        if any(np.linalg.norm(point.xyz_RM - old.xyz_RM) < spacing for old in chosen):
            continue
        chosen.append(point)

    # Tail reconnection X-points remain first.  Within either group, choose
    # the better of the two slice coordinates for a monotonic chain; only an
    # overlapping/ambiguous projection falls back to X/O-aware nearest links.
    tail_x = [p for p in chosen if p.kind == "X" and p.xyz_RM[0] < 0.0]
    dayside = [p for p in chosen if p.xyz_RM[0] >= 0.0]
    return _order_plane_group(tail_x, tangent) + _order_plane_group(dayside, tangent)
