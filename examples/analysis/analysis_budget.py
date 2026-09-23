"""Time-series Na+ inventory derivative and conservation-budget diagnostics."""

from __future__ import annotations

import numpy as np


BUDGET_UNITS = {
    "Na_inventory_time_derivative_particles_s": "particles/s",
    "Na_budget_residual_particles_s": "particles/s",
}


def add_budget_columns(rows: list[dict[str, float]]) -> list[dict[str, float]]:
    """Add ``dot(N_Na+)`` and the requested Na+ budget residual.

    ``dot(N)`` uses a first-order one-sided derivative at each sequence end
    and a centred, nonuniform-time finite difference in the interior.
    Closure uses the signed net outward flux through every control-volume
    boundary, not the one-way escape diagnostic:
    ``dot(N) - Q_src + Phi_surface_net + Phi_outer_net``.
    """
    required = (
        "time",
        "Na_inventory_particles",
        "Na_photoionization_source_particles_s",
        "Na_surface_net_outward_particles_s",
        "Na_outer_boundary_net_outward_particles_s",
    )
    if any(any(name not in row for name in required) for row in rows):
        for row in rows:
            row["Na_inventory_time_derivative_particles_s"] = float("nan")
            row["Na_budget_residual_particles_s"] = float("nan")
        return rows
    time = np.asarray([row["time"] for row in rows], dtype=float)
    inventory = np.asarray([row["Na_inventory_particles"] for row in rows], dtype=float)
    derivative = np.full(time.shape, np.nan, dtype=float)
    if time.size >= 2 and np.all(np.isfinite(time)) and np.all(np.isfinite(inventory)) and np.all(np.diff(time) > 0.0):
        derivative = np.gradient(inventory, time, edge_order=1)
    for index, row in enumerate(rows):
        source = float(row["Na_photoionization_source_particles_s"])
        surface_net = float(row["Na_surface_net_outward_particles_s"])
        outer_net = float(row["Na_outer_boundary_net_outward_particles_s"])
        row["Na_inventory_time_derivative_particles_s"] = float(derivative[index])
        row["Na_budget_residual_particles_s"] = float(
            derivative[index] - source + surface_net + outer_net
        )
    return rows
