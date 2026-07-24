"""O-point search and local LMN Tecplot diagnostics.

An O-point is identified from a least-squares gradient of the *unit* total
magnetic field.  Its gradient must have one real eigenvalue and one conjugate
complex pair, while the unit field is parallel to the real eigenvector.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from mpcns_post.errors import ValidationError
from mpcns_post.tecplot import TecplotZone, inspect_tecplot_binary, write_tecplot_binary


# None searches every Fluid Cell.  Set ((xmin,xmax),...) to restrict a run.
SEARCH_BOX_RM = None
# Keep the fit local to the dayside current sheet; 18 is deliberately within
# the requested 6--27 range and avoids blending opposite sides of the sheet.
DONOR_NEIGHBORS = 27
JMAG_MIN_NA_M2 = 400.0
MAX_DONOR_RADIUS_RM = None       # None: six median Cell-centre spacings
MAX_REDUCED_FIELD = 0.25         # |b - (b.er) er|; smaller is stricter
MIN_COMPLEX_IMAGINARY = 1.0e-7   # relative to ||grad(b)||
MIN_OPOINT_SEPARATION_RM = 0.025
MAX_OPOINTS = 80

# A true transverse O-point has one positive phase winding of B_L+i B_N.
# Radii are in locally derived median Cell-centre spacings, keeping this test
# local even where the structured grid is stretched.
WINDING_RADIUS_IN_SPACINGS = (2.0, 3.0, 4.0)
WINDING_SAMPLES = 96
WINDING_TARGET = 1.0
WINDING_TOLERANCE = 0.15
MIN_CIRCLE_TRANSVERSE_B_NT = 1.0e-5

# One or more L-N planes are output for every accepted O-point.
SLICE_M_OFFSETS_RM = (0.0,)
SLICE_L_RANGE_RM = (-0.15, 0.15)
SLICE_N_RANGE_RM = (-0.15, 0.15)
SLICE_NL = 20
SLICE_NN = 20
INTERPOLATION_NEIGHBORS = 8
MAX_INTERPOLATION_DISTANCE_RM = None

# O-points are grouped into spatially connected, field-aligned components and
# each component is represented by a polynomial regression curve.
CURVE_LINK_DISTANCE_RM = 0.08
CURVE_MIN_ALIGNMENT = 0.75
MIN_POINTS_PER_REGRESSION = 3
REGRESSION_SAMPLES = 101

MU0_H_M = 4.0e-7 * np.pi


def _unit(v, name):
    v = np.asarray(v, dtype=float)
    norm = float(np.linalg.norm(v))
    if not np.isfinite(norm) or norm <= 1.0e-14:
        raise ValidationError(f"Cannot normalize {name}: magnitude is {norm}")
    return v / norm


def _expand(fields):
    out = {}
    for name, value in fields.items():
        value = np.asarray(value)
        if value.ndim == 1:
            out[name] = value
        elif value.ndim == 2 and value.shape[1] == 3:
            for i, component in enumerate("xyz"):
                out[f"{name}_{component}"] = value[:, i]
        else:
            raise ValueError(f"{name}: expected (N,) or (N,3), got {value.shape}")
    return out


class _Interpolator:
    def __init__(self, xyz, neighbors, max_distance):
        self.xyz = np.asarray(xyz, dtype=float)
        self.tree = cKDTree(self.xyz)
        self.neighbors = min(max(1, int(neighbors)), len(self.xyz))
        if max_distance is None:
            d = self.tree.query(self.xyz, k=2)[0][:, 1]
            max_distance = 3.0 * float(np.median(d[np.isfinite(d) & (d > 0)]))
        self.max_distance = float(max_distance)

    def sample(self, points, values):
        points = np.asarray(points, dtype=float)
        shape = points.shape[:-1]
        distance, index = self.tree.query(points.reshape(-1, 3), k=self.neighbors)
        if self.neighbors == 1:
            distance, index = distance[:, None], index[:, None]
        valid = np.isfinite(distance[:, 0]) & (distance[:, 0] <= self.max_distance)
        weight = 1.0 / np.maximum(distance, 1.0e-14) ** 2
        values = np.asarray(values)[index]
        if values.ndim == 2:
            result = (weight * values).sum(1) / weight.sum(1)
        else:
            result = (weight[..., None] * values).sum(1) / weight.sum(1)[:, None]
        result[~valid] = np.nan
        return result.reshape(shape + result.shape[1:]), valid.reshape(shape)


def _unit_field_gradient(xyz, b, tree, index, donor_radius):
    """Return G where delta_b = G @ delta_x, fitted from donor Cells."""
    distance, donors = tree.query(xyz[index], k=min(DONOR_NEIGHBORS, len(xyz)))
    keep = (distance > 1.0e-12) & (distance <= donor_radius)
    donors = np.asarray(donors)[keep]
    if donors.size < 6:
        return None
    dx, db = xyz[donors] - xyz[index], b[donors] - b[index]
    if np.linalg.matrix_rank(dx) < 3:
        return None
    # lstsq solves dx @ A = db, hence G = A.T.
    return np.linalg.lstsq(dx, db, rcond=None)[0].T


def _find_opoints(xyz, B, b, grad_p, candidate, donor_radius):
    tree = cKDTree(xyz)
    found = []
    for index in candidate:
        G = _unit_field_gradient(xyz, b, tree, int(index), donor_radius)
        if G is None or not np.all(np.isfinite(G)):
            continue
        eigval, eigvec = np.linalg.eig(G)
        scale = max(float(np.linalg.norm(G)), 1.0e-14)
        real = np.flatnonzero(np.abs(eigval.imag) <= 1.0e-9 * scale)
        complex_pair = np.flatnonzero(np.abs(eigval.imag) > MIN_COMPLEX_IMAGINARY * scale)
        if real.size != 1 or complex_pair.size != 2:
            continue
        er = _unit(eigvec[:, real[0]].real, "real eigenvector of grad(b)")
        if np.dot(er, b[index]) < 0:
            er = -er
        w = b[index] - np.dot(b[index], er) * er
        reduced = float(np.linalg.norm(w))
        if reduced > MAX_REDUCED_FIELD:
            continue
        try:
            M = b[index]
            N0 = _unit(grad_p[index], "total-pressure gradient")
            # Orthogonalize N against M so LMN is a genuine right-handed basis.
            N = _unit(N0 - np.dot(N0, M) * M, "pressure-gradient component perpendicular to M")
            L = _unit(np.cross(M, N), "M cross N")
        except ValidationError:
            continue
        R = np.column_stack((L, M, N))
        Q = R.T @ G @ R
        G_B = _unit_field_gradient(xyz, B, tree, int(index), donor_radius)
        if G_B is None or not np.all(np.isfinite(G_B)):
            continue
        # This is the requested physical B-component Jacobian, whereas Q is
        # retained for the unit-B topology classification.
        LN = (R.T @ G_B @ R)[np.ix_((0, 2), (0, 2))]
        trace, determinant = float(np.trace(LN)), float(np.linalg.det(LN))
        discriminant = trace * trace - 4.0 * determinant
        found.append(dict(index=int(index), xyz=xyz[index], b=M, L=L, M=M, N=N,
                          G=G, G_B=G_B, eigval=eigval, er=er, reduced=reduced, Q=Q, LN=LN,
                          trace=trace, determinant=determinant, discriminant=discriminant))
    # Retain the best aligned representatives rather than many adjacent Cells.
    found.sort(key=lambda item: item["reduced"])
    chosen = []
    for item in found:
        if all(np.linalg.norm(item["xyz"] - other["xyz"]) >= MIN_OPOINT_SEPARATION_RM for other in chosen):
            chosen.append(item)
        if len(chosen) >= MAX_OPOINTS:
            break
    return chosen


def _transverse_winding(point, interpolator, B, radii):
    """Phase winding of B_L+iB_N on fixed-LMN circular contours."""
    angle = np.linspace(0.0, 2.0 * np.pi, WINDING_SAMPLES, endpoint=False)
    windings = []
    for radius in radii:
        circle = point["xyz"] + radius * (
            np.cos(angle)[:, None] * point["L"] + np.sin(angle)[:, None] * point["N"]
        )
        field, valid = interpolator.sample(circle, B)
        if not np.all(valid) or not np.all(np.isfinite(field)):
            return None
        transverse = (field @ point["L"]) + 1j * (field @ point["N"])
        if np.min(np.abs(transverse)) <= MIN_CIRCLE_TRANSVERSE_B_NT:
            return None
        phase = np.unwrap(np.angle(np.r_[transverse, transverse[0]]))
        windings.append(float((phase[-1] - phase[0]) / (2.0 * np.pi)))
    return np.asarray(windings)


def _regression_curves(points):
    """Fit field-aligned polynomial centre curves and return local tangents."""
    count = len(points)
    if not count:
        return [], np.full((0,), -1), np.full((0, 3), np.nan)
    xyz = np.array([p["xyz"] for p in points])
    direction = np.array([p["M"] for p in points])
    tree = cKDTree(xyz)
    graph = [set() for _ in points]
    for i, neighbours in enumerate(tree.query_ball_point(xyz, CURVE_LINK_DISTANCE_RM)):
        for j in neighbours:
            if j <= i:
                continue
            alignment = abs(float(np.dot(direction[i], direction[j])))
            displacement = _unit(xyz[j] - xyz[i], "O-point separation")
            if alignment >= CURVE_MIN_ALIGNMENT and max(abs(np.dot(displacement, direction[i])), abs(np.dot(displacement, direction[j]))) >= 0.25:
                graph[i].add(j); graph[j].add(i)
    components, seen = [], set()
    for root in range(count):
        if root in seen:
            continue
        todo, component = [root], []
        seen.add(root)
        while todo:
            i = todo.pop(); component.append(i)
            for j in graph[i]:
                if j not in seen: seen.add(j); todo.append(j)
        if len(component) >= MIN_POINTS_PER_REGRESSION:
            components.append(component)
    curve_id, tangents, curves = np.full(count, -1), np.full((count, 3), np.nan), []
    for cid, component in enumerate(components):
        coords = xyz[component]
        centre = coords.mean(0)
        _, _, vh = np.linalg.svd(coords - centre, full_matrices=False)
        axis = vh[0]
        if np.mean(direction[component] @ axis) < 0: axis = -axis
        s = (coords - centre) @ axis
        degree = min(3, len(component) - 1)
        coefficients = [np.polyfit(s, coords[:, k], degree) for k in range(3)]
        ss = np.linspace(s.min(), s.max(), REGRESSION_SAMPLES)
        curve = np.column_stack([np.polyval(c, ss) for c in coefficients])
        derivative = np.column_stack([np.polyval(np.polyder(c), s) for c in coefficients])
        derivative = np.array([_unit(v, "regression tangent") for v in derivative])
        for local, global_index in enumerate(component):
            curve_id[global_index], tangents[global_index] = cid, derivative[local]
        curves.append((cid, curve))
    return curves, curve_id, tangents


def export_flux_rope(data: dict) -> dict:
    """Search O-points, emit their L-N slices, coordinates, curves and report."""
    case, arrays = data["case"], data["CELL_ARRAYS"]
    fluid = np.asarray(data["fluid_mask"], bool)
    xyz_all, B_all = np.asarray(case.cells.coordinates, float), np.asarray(arrays["B_total_nT"], float)
    bnorm = np.linalg.norm(B_all, axis=1)
    p_total = (np.asarray(arrays["H_pressure_nPa"], float) + np.asarray(arrays["Na_pressure_nPa"], float)
               + np.asarray(arrays["electron_pressure_Pa"], float) * 1e9 + np.sum(B_all**2, axis=1) * 1e-9 / (2 * MU0_H_M))
    p_scale = np.nanmax(np.abs(p_total[fluid]))
    grad_p_all = data["compute_gradient_on_fluid_cells"](case, p_total / p_scale)
    J_all = np.asarray(arrays["J_induced_nA_m2"], float)
    jmag_all = np.linalg.norm(J_all, axis=1)
    valid = fluid & np.isfinite(bnorm) & (bnorm > 1e-12) & np.isfinite(p_total) & np.all(np.isfinite(grad_p_all), axis=1)
    xyz, B, b, grad_p = xyz_all[valid], B_all[valid], B_all[valid] / bnorm[valid, None], grad_p_all[valid]
    if len(xyz) < DONOR_NEIGHBORS:
        raise ValidationError("Too few valid Fluid Cells for O-point donor fit")
    # The donor cloud may include all valid Fluid Cells, but only the strong
    # current-sheet Cells are eligible to become O-point centres.
    strong_current = jmag_all[valid] > JMAG_MIN_NA_M2
    if SEARCH_BOX_RM is None:
        candidate = np.flatnonzero(strong_current)
    else:
        bounds = np.asarray(SEARCH_BOX_RM, float)
        if bounds.shape != (3, 2): raise ValueError("SEARCH_BOX_RM must be None or ((xmin,xmax),...)")
        candidate = np.flatnonzero(strong_current & np.all((xyz >= bounds[:, 0]) & (xyz <= bounds[:, 1]), axis=1))
    if candidate.size == 0:
        raise ValidationError(f"No valid Fluid Cell has |J_induced| > {JMAG_MIN_NA_M2:g} nA/m^2 in the search region")
    tree = cKDTree(xyz)
    spacing = np.median(tree.query(xyz, k=2)[0][:, 1])
    donor_radius = float(MAX_DONOR_RADIUS_RM or 6.0 * spacing)
    points = _find_opoints(xyz, B, b, grad_p, candidate, donor_radius)
    if not points:
        raise ValidationError("No O-points met the eigenvalue and reduced-field criteria; relax MAX_REDUCED_FIELD or enlarge SEARCH_BOX_RM")
    interpolator = _Interpolator(xyz, INTERPOLATION_NEIGHBORS, MAX_INTERPOLATION_DISTANCE_RM)
    winding_radii = spacing * np.asarray(WINDING_RADIUS_IN_SPACINGS, dtype=float)
    spectral_count = len(points)
    winding_points = []
    for point in points:
        winding = _transverse_winding(point, interpolator, B, winding_radii)
        if winding is not None and np.all(np.abs(winding - WINDING_TARGET) <= WINDING_TOLERANCE):
            point["winding"] = winding
            winding_points.append(point)
    points = winding_points
    if not points:
        raise ValidationError(
            "No spectral candidate has +1 transverse-field winding at every configured radius; "
            "inspect WINDING_RADIUS_IN_SPACINGS or WINDING_TOLERANCE"
        )
    curves, curve_ids, tangents = _regression_curves(points)
    for i, point in enumerate(points):
        point["curve_id"] = int(curve_ids[i])
        point["curve_tangent"] = tangents[i]
        point["curve_cosine"] = float(abs(np.dot(point["M"], tangents[i]))) if curve_ids[i] >= 0 else np.nan
    output = Path(data["OUTPUT_DIR"]) / "flux_rope"
    output.mkdir(parents=True, exist_ok=True)

    # Coordinate table and human-readable diagnostic report.
    with (output / "o_points.csv").open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp); writer.writerow(["id", "x_RM", "y_RM", "z_RM", "curve_id", "reduced_field", "curve_cosine", *[f"winding_r{r:.6g}RM" for r in winding_radii]])
        for i, p in enumerate(points): writer.writerow([i, *p["xyz"], p["curve_id"], p["reduced"], p["curve_cosine"], *p["winding"]])
    report = ["O-point diagnostic", f"count = {len(points)}", f"candidate_count = {candidate.size}", f"spectral_candidate_count = {spectral_count}", f"Jmag_min_nA_m2 = {JMAG_MIN_NA_M2:.8g}", f"donor_neighbors = {DONOR_NEIGHBORS}", f"donor_radius_RM = {donor_radius:.8g}", f"winding_radii_RM = {winding_radii.tolist()}", f"winding_target_tolerance = {WINDING_TARGET:.8g}, {WINDING_TOLERANCE:.8g}", ""]
    for i, p in enumerate(points):
        report += [f"O-point {i}: x_RM = {p['xyz'].tolist()}", f"  L = {p['L'].tolist()}", f"  M = {p['M'].tolist()}", f"  N = {p['N'].tolist()}",
                   f"  B_L(x0), B_N(x0) [nT] = {float(B[p['index']] @ p['L']):.8g}, {float(B[p['index']] @ p['N']):.8g}",
                   f"  d(BL,BN)/d(L,N) [nT/RM] = {p['LN'].tolist()}", f"  det = {p['determinant']:.8g}; trace = {p['trace']:.8g}; trace^2-4det = {p['discriminant']:.8g}",
                   f"  grad(unit-B) = {p['G'].tolist()}", f"  eig(grad unit-B) = {p['eigval'].tolist()}", f"  winding(BL+iBN) = {p['winding'].tolist()}", f"  |w| = {p['reduced']:.8g}; curve_id = {p['curve_id']}; |M.dot(t_curve)| = {p['curve_cosine']:.8g}", ""]
    (output / "o_points_diagnostic.txt").write_text("\n".join(report), encoding="utf-8")

    zones = []
    la, na = np.linspace(*SLICE_L_RANGE_RM, SLICE_NL), np.linspace(*SLICE_N_RANGE_RM, SLICE_NN)
    ll, nn = np.meshgrid(la, na, indexing="ij")
    for oid, p in enumerate(points):
        for moffset in SLICE_M_OFFSETS_RM:
            pos = p["xyz"] + moffset * p["M"] + ll[..., None] * p["L"] + nn[..., None] * p["N"]
            flat = pos.reshape(-1, 3)
            Bp, covered = interpolator.sample(flat, B)
            if not np.all(covered):
                continue
            pp, _ = interpolator.sample(flat, p_total[valid])
            bp = Bp / np.linalg.norm(Bp, axis=1)[:, None]
            fields = {"x_RM": flat[:, 0], "y_RM": flat[:, 1], "z_RM": flat[:, 2], "l_RM": ll.ravel(), "m_RM": np.full(len(flat), moffset), "n_RM": nn.ravel(),
                      "B_total_nT": Bp, "b_unit": bp, "total_pressure_nPa": pp, "B_L_nT": Bp @ p["L"], "B_M_nT": Bp @ p["M"], "B_N_nT": Bp @ p["N"]}
            values = {name: np.asarray(value).reshape((SLICE_NL, SLICE_NN, 1)) for name, value in _expand(fields).items()}
            zones.append(TecplotZone(f"opoint_{oid:03d}_LN_m{moffset:+.5f}", "Fluid", values))
    if not zones: raise ValidationError("All requested O-point slices are outside interpolation coverage")
    slice_path = write_tecplot_binary(output / "o_points_LN_slices.plt", title="MPCNS O-point LMN slices", variable_names=tuple(zones[0].values), zones=zones, solution_time=float(case.latest_restart[0].time))
    curve_zones = []
    for cid, curve in curves:
        values = {"x_RM": curve[:, 0].reshape((-1, 1, 1)), "y_RM": curve[:, 1].reshape((-1, 1, 1)), "z_RM": curve[:, 2].reshape((-1, 1, 1)), "curve_id": np.full((len(curve), 1, 1), cid)}
        curve_zones.append(TecplotZone(f"opoint_regression_curve_{cid:03d}", "Fluid", values))
    curve_path = None if not curve_zones else write_tecplot_binary(output / "o_point_regression_curves.plt", title="O-point regression curves", variable_names=tuple(curve_zones[0].values), zones=curve_zones, solution_time=float(case.latest_restart[0].time))
    summary = {"o_point_count": len(points), "slice_file": str(slice_path), "curve_file": None if curve_path is None else str(curve_path), "coordinate_file": str(output / "o_points.csv"), "diagnostic_file": str(output / "o_points_diagnostic.txt")}
    (output / "o_points_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("Written O-point slices:", inspect_tecplot_binary(slice_path).path)
    print("O-points:", len(points), "regression curves:", len(curves))
    return summary
