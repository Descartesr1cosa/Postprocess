"""Structured-grid X/O-point search on one Cartesian slice.

Topology is always detected from existing Cell-centred fields on structured
four-Cell quadrilaterals. Coordinates can be grid-locked Cell centres or local
affine zero estimates within an accepted quadrilateral.
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
        pairwise = np.linalg.norm(coordinates[:, None] - coordinates[None, :], axis=2)
        spacing = float(np.median(pairwise[np.triu_indices(len(points), k=1)]))
        if np.min(projected_gap) >= 0.25 * spacing:
            path = float(np.sum(np.linalg.norm(np.diff(coordinates[order], axis=0), axis=1)))
            candidates.append((path, order))
    if candidates:
        _, order = min(candidates, key=lambda item: item[0])
        return [points[index] for index in order]
    return _nearest_type_chain(points)


def _slice_indices(coordinates: np.ndarray, *, normal_axis: int, value_rm: float,
                   tolerance_rm: float) -> list[tuple[int, int]]:
    """Find logical-index slices coincident with the requested Cartesian plane."""
    matches = []
    for logical_axis in range(3):
        normal_coordinate = np.moveaxis(coordinates[..., normal_axis], logical_axis, 0)
        reduce_axes = tuple(range(1, normal_coordinate.ndim))
        distance = np.max(np.abs(normal_coordinate - value_rm), axis=reduce_axes)
        for index in np.flatnonzero(distance <= tolerance_rm):
            matches.append((logical_axis, int(index)))
    return matches


def _quadrilateral_topology(xyz: np.ndarray, b: np.ndarray, gids: np.ndarray,
                            *, normal_axis: int, position_mode: str,
                            plane_value_rm: float) -> list[tuple[float, int, int, float, PlanePoint]]:
    """Detect topological zeros from four neighbouring Cell centres."""
    tangent = _plane_axes(normal_axis)
    # For y=y0 this is (z,x) for both coordinates and field components.
    planar_axes = tuple(reversed(tangent))
    shape = xyz.shape[:2]
    found = []
    for i in range(shape[0] - 1):
        for j in range(shape[1] - 1):
            corners = np.array(((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)))
            positions = xyz[corners[:, 0], corners[:, 1]][:, planar_axes]
            values = b[corners[:, 0], corners[:, 1]][:, planar_axes]
            if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(values)):
                continue
            centre = positions.mean(axis=0)
            angle = np.arctan2(positions[:, 1] - centre[1], positions[:, 0] - centre[0])
            order = np.argsort(angle)
            phase = np.angle(values[order, 0] + 1j * values[order, 1])
            winding = float(np.sum(np.angle(np.exp(1j * np.diff(np.r_[phase, phase[0]])))) / (2.0 * np.pi))
            if abs(abs(winding) - 1.0) > 0.25:
                continue
            # Least-squares local Jacobian, matching post_flux_rope's
            # trace/determinant/discriminant topology classification.
            dx = positions - centre
            db = values - values.mean(axis=0)
            if np.linalg.matrix_rank(dx) < 2:
                continue
            jacobian = np.linalg.lstsq(dx, db, rcond=None)[0].T
            trace = float(np.trace(jacobian))
            determinant = float(np.linalg.det(jacobian))
            discriminant = trace * trace - 4.0 * determinant
            if determinant < 0.0 and discriminant > 0.0:
                kind, expected_winding = "X", -1.0
            elif determinant > 0.0 and discriminant < 0.0:
                kind, expected_winding = "O", 1.0
            else:
                continue
            if abs(winding - expected_winding) > 0.25:
                continue
            magnitude = np.linalg.norm(values, axis=1)
            local = int(np.argmin(magnitude))
            ci, cj = corners[local]
            gid = int(gids[ci, cj])
            if position_mode == "cell_center":
                point_xyz = xyz[ci, cj].copy()
            else:
                # B_t ~= mean(B_t) + J (r_t - mean(r_t)); this only locates
                # the already accepted topology and does not reclassify it.
                root_planar = centre - np.linalg.solve(jacobian, values.mean(axis=0))
                point_xyz = xyz[corners[:, 0], corners[:, 1]].mean(axis=0)
                point_xyz[list(planar_axes)] = root_planar
                point_xyz[normal_axis] = plane_value_rm
            # Prefer the representative whose in-plane field is smallest.
            charge = -1 if kind == "X" else 1
            edge_length = np.linalg.norm(positions - np.roll(positions, -1, axis=0), axis=1)
            local_spacing = float(np.median(edge_length))
            found.append((float(magnitude[local]), gid, charge, local_spacing, PlanePoint(kind, point_xyz)))
    return found


def _merge_nearby_topology(raw: list[tuple[float, int, int, float, PlanePoint]], *,
                           radius_in_spacings: float) -> list[PlanePoint]:
    """Merge repeated structured-quadrilateral contours of the same zero."""
    raw.sort(key=lambda item: item[0])
    by_gid = []
    seen_gids = set()
    for item in raw:
        if item[1] not in seen_gids:
            by_gid.append(item)
            seen_gids.add(item[1])
    if not by_gid:
        return []
    xyz = np.asarray([item[4].xyz_RM for item in by_gid])
    local_scale = np.asarray([item[3] for item in by_gid])
    tree = cKDTree(xyz)
    unseen = set(range(len(by_gid)))
    merged = []
    while unseen:
        root = unseen.pop()
        component = {root}
        todo = [root]
        while todo:
            index = todo.pop()
            maximum_radius = radius_in_spacings * max(local_scale[index], float(np.max(local_scale)))
            neighbours = {
                other for other in tree.query_ball_point(xyz[index], maximum_radius)
                if other in unseen
                and np.linalg.norm(xyz[other] - xyz[index])
                <= radius_in_spacings * max(local_scale[index], local_scale[other])
            }
            unseen -= neighbours
            component |= neighbours
            todo.extend(neighbours)
        entries = [by_gid[index] for index in component]
        charge = sum(item[2] for item in entries)
        # A zero net winding is possible when adjacent discrete contours have
        # opposite signs.  In that tie, retain the lowest-|B| representative
        # instead of silently losing a physically visible cell-centred point.
        kind = "X" if charge < 0 else "O" if charge > 0 else min(entries, key=lambda item: item[0])[4].kind
        compatible = [item for item in entries if item[4].kind == kind]
        merged.append(min(compatible or entries, key=lambda item: item[0])[4])
    return merged


def find_plane_xo_points(case, B_total: np.ndarray, fluid_mask: np.ndarray, *,
                         normal_axis: str, value_rm: float, tolerance_rm: float,
                         merge_radius_in_spacings: float = 3.0,
                         position_mode: str = "cell_center") -> list[PlanePoint]:
    """Find X/O points on an x/y/z=value mesh slice.

    A matching logical slice is identified in every Fluid structured block.
    The method needs the case's block maps, rather than a global point cloud,
    so each winding contour follows genuine two-dimensional mesh neighbours.
    ``position_mode`` is ``cell_center`` or ``interpolated``.
    """
    axis = {"x": 0, "y": 1, "z": 2}.get(normal_axis.lower())
    if axis is None:
        raise ValueError("normal_axis must be 'x', 'y', or 'z'")
    if tolerance_rm < 0.0:
        raise ValueError("tolerance_rm must be non-negative")
    if position_mode not in {"cell_center", "interpolated"}:
        raise ValueError("position_mode must be 'cell_center' or 'interpolated'")
    xyz_all = np.asarray(case.cells.coordinates, dtype=float)
    b_all = np.asarray(B_total, dtype=float)
    fluid_all = np.asarray(fluid_mask, dtype=bool)
    gid_all = np.asarray(case.cells.global_ids)
    raw = []
    slices = 0
    for block in case.iter_blocks(location="cell"):
        if block.physics != "Fluid":
            continue
        xyz = block.reshape(xyz_all)
        b = block.reshape(b_all)
        fluid = block.reshape(fluid_all)
        gids = block.reshape(gid_all)
        for logical_axis, fixed_index in _slice_indices(
            xyz, normal_axis=axis, value_rm=value_rm, tolerance_rm=tolerance_rm,
        ):
            slice_xyz = np.take(xyz, fixed_index, axis=logical_axis)
            slice_b = np.take(b, fixed_index, axis=logical_axis)
            slice_fluid = np.take(fluid, fixed_index, axis=logical_axis)
            slice_gids = np.take(gids, fixed_index, axis=logical_axis)
            if min(slice_xyz.shape[:2]) < 2:
                continue
            # Reject quadrilaterals touching non-fluid Cells.
            slice_b = slice_b.copy()
            slice_b[~slice_fluid] = np.nan
            raw.extend(_quadrilateral_topology(
                slice_xyz, slice_b, slice_gids,
                normal_axis=axis, position_mode=position_mode,
                plane_value_rm=value_rm,
            ))
            slices += 1
    if not slices:
        raise ValueError(
            "No structured Cell slice matches the requested plane; increase PLANE_TOLERANCE_RM "
            "or choose a mesh-represented symmetry plane"
        )
    chosen = _merge_nearby_topology(
        raw, radius_in_spacings=merge_radius_in_spacings,
    )

    tangent = _plane_axes(axis)
    tail_x = [point for point in chosen if point.kind == "X" and point.xyz_RM[0] < 0.0]
    dayside = [point for point in chosen if point.xyz_RM[0] >= 0.0]
    return _order_plane_group(tail_x, tangent) + _order_plane_group(dayside, tangent)
