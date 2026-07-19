"""Vectorized spatial selections over globally assembled entity coordinates."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .access import Selection, base_location


AXES = {"x": 0, "y": 1, "z": 2, 0: 0, 1: 1, 2: 2}


def _selection(case, location: str, mask: np.ndarray) -> Selection:
    entities = case.entity(location)
    selected = np.asarray(mask, dtype=bool)
    if selected.shape != (entities.size,):
        raise ValueError(
            f"selection mask must have shape ({entities.size},), got {selected.shape}"
        )
    indices = np.flatnonzero(selected)
    return Selection(
        base_location(location),
        indices,
        entities.coordinates[indices],
        entities.global_ids[indices],
        selected,
    )


def select_plane(
    case,
    *,
    axis: str | int,
    value: float,
    tolerance: float = 0.0,
    location: str = "cell",
) -> Selection:
    """Select entities whose coordinate lies within a Cartesian plane slab."""
    try:
        component = AXES[axis]
    except KeyError as exc:
        raise ValueError("axis must be x/y/z or 0/1/2") from exc
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    coordinates = case.entity(location).coordinates[:, component]
    mask = np.abs(coordinates - float(value)) <= tolerance
    return _selection(case, location, mask)


def select_box(
    case,
    lower: Sequence[float] | None = None,
    upper: Sequence[float] | None = None,
    *,
    x: tuple[float, float] | None = None,
    y: tuple[float, float] | None = None,
    z: tuple[float, float] | None = None,
    location: str = "cell",
) -> Selection:
    """Select an axis-aligned box using corners or x/y/z interval keywords."""
    if lower is not None or upper is not None:
        if lower is None or upper is None or any(interval is not None for interval in (x, y, z)):
            raise ValueError("provide both lower/upper corners or x/y/z intervals")
        lo = np.asarray(lower, dtype=np.float64)
        hi = np.asarray(upper, dtype=np.float64)
    else:
        intervals = (
            (-np.inf, np.inf) if x is None else x,
            (-np.inf, np.inf) if y is None else y,
            (-np.inf, np.inf) if z is None else z,
        )
        lo = np.asarray([interval[0] for interval in intervals], dtype=np.float64)
        hi = np.asarray([interval[1] for interval in intervals], dtype=np.float64)
    if lo.shape != (3,) or hi.shape != (3,) or np.any(lo > hi):
        raise ValueError("box bounds must be ordered three-component coordinates")
    coordinates = case.entity(location).coordinates
    mask = np.all((coordinates >= lo) & (coordinates <= hi), axis=1)
    return _selection(case, location, mask)


def select_sphere(
    case,
    *,
    center: Sequence[float] = (0.0, 0.0, 0.0),
    radius: float,
    inner_radius: float = 0.0,
    location: str = "cell",
) -> Selection:
    """Select a sphere or spherical shell in the stored coordinate system."""
    origin = np.asarray(center, dtype=np.float64)
    if origin.shape != (3,) or radius < 0 or inner_radius < 0 or inner_radius > radius:
        raise ValueError("invalid sphere center/radius")
    distance = np.linalg.norm(case.entity(location).coordinates - origin, axis=1)
    mask = (distance >= inner_radius) & (distance <= radius)
    return _selection(case, location, mask)
