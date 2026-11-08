"""Offline quality checks for saved PathCover polynomials and yaw arrays.

The polynomial extrema, one-sided saved-yaw evaluation, local rate refinement,
and acceptance allowances are the previously validated formal-run algorithms.
No current frontend defaults, native extension, optimizer, or GPU are used.
Polynomial roots and sampled/local rate refinement are numerical diagnostics,
not interval certificates of global feasibility. Mission success and measured
vehicle/MPC constraints remain separate from returned-reference quality.
"""
from __future__ import annotations

import hashlib
import json
import numbers
from collections import Counter
from pathlib import Path

import numpy as np
from numpy.polynomial import Polynomial
from scipy.optimize import minimize_scalar

RATE_NAMES = ("abs_body_p", "abs_body_q", "abs_body_r", "body_rate_norm",
              "abs_world_yaw_rate", "zero_yaw_proxy_norm")


def initial_boundary_contract(curve, state, acceleration, config, recorded=None):
    """Compare the saved initial curve with its recorded boundary contract.

    Missing historical metadata means fixed PVA. No current factory default is
    consulted, and a free initial collective does not impose measured a0.
    """
    metadata = {} if recorded is None else dict(recorded)
    mode = "fixed_pva" if recorded is None else metadata.get("mode")
    if mode not in ("fixed_pva", "ctbr_collective"):
        raise ValueError(f"Unknown saved initial boundary mode: {mode!r}")
    state = np.asarray(state, dtype=np.float64).reshape(10)
    measured_a = np.asarray(acceleration, dtype=np.float64).reshape(3)
    measured = np.column_stack((state[:3], state[7:10], measured_a))
    actual = np.column_stack([value[0] for value in curve.derivatives([0.0])[:3]])
    difference = actual - measured
    quaternion = state[3:7]
    quaternion_norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(quaternion_norm) or quaternion_norm == 0:
        raise ValueError("Saved initial measurement has an invalid quaternion")
    w, x, y, z = quaternion / quaternion_norm
    rotation = np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y)],
        [2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x)],
        [2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y)],
    ])
    axis = rotation[:, 2]
    frame_heading = float(np.arctan2(-rotation[0, 1], rotation[1, 1]))
    yaw = float(curve.yaw([0.0])[0][0])
    gravity = np.array([0.0, 0.0, float(config["gravity"])])
    force_vector = actual[:, 2] + gravity
    specific_force = float(np.linalg.norm(force_vector))
    force_n = float(config["mass"]) * specific_force
    frame_error = None
    reference_axis = None
    if specific_force > 0 and force_vector[2] > 0:
        reference_axis = force_vector / specific_force
        heading = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        reference_y = np.cross(reference_axis, heading)
        reference_y /= np.linalg.norm(reference_y)
        reference_rotation = np.column_stack((np.cross(reference_y, reference_axis), reference_y, reference_axis))
        relative = rotation.T @ reference_rotation
        sine = 0.5 * np.linalg.norm([relative[2, 1] - relative[1, 2],
                                     relative[0, 2] - relative[2, 0],
                                     relative[1, 0] - relative[0, 1]])
        frame_error = float(np.arctan2(sine, 0.5 * (np.trace(relative) - 1.0)))
    consistency = {}
    for key, expected in (("measured_acceleration", measured_a),
                          ("reference_acceleration", actual[:, 2]), ("thrust_axis", axis)):
        value = metadata.get(key)
        consistency[key + "_max_abs_difference"] = (None if value is None else
            float(np.max(np.abs(np.asarray(value, dtype=float) - expected))))
    value = metadata.get("force_n")
    consistency["force_n_abs_difference"] = None if value is None else abs(float(value) - force_n)
    for key, expected in (("force_lower_n", config["thrust_min"]), ("force_upper_n", config["thrust_max"])):
        value = metadata.get(key)
        consistency[key + "_abs_difference"] = None if value is None else abs(float(value) - float(expected))
    nominal = config.get("nominal_thrust_min")
    nominal = float(config["thrust_min"] if nominal is None else nominal)
    measured_force = float(config["mass"] * np.linalg.norm(measured_a + gravity))
    return {
        "mode": mode, "metadata_present": recorded is not None, "recorded": metadata,
        "mode_source": "recorded" if recorded is not None else "historical absence means fixed_pva",
        "measured_initial_pva": measured, "reference_initial_pva": actual,
        "reference_minus_measured_initial_pva": difference,
        "maximum_absolute_difference_by_pva_column": np.max(np.abs(difference), axis=0),
        "fixed_position_max_abs_error_m": float(np.max(np.abs(difference[:, 0]))),
        "fixed_velocity_max_abs_error_m_s": float(np.max(np.abs(difference[:, 1]))),
        "fixed_acceleration_max_abs_error_m_s2": (float(np.max(np.abs(difference[:, 2])))
                                                  if mode == "fixed_pva" else None),
        "measured_quaternion_norm": quaternion_norm,
        "measured_body_z": axis, "reference_body_z": reference_axis,
        "measured_frame_heading_rad": frame_heading, "reference_heading_rad": yaw,
        "reference_measured_rotation_difference_rad": frame_error,
        "fixed_attitude_rotation_error_rad": frame_error if mode == "ctbr_collective" else None,
        "reference_upright_frame_defined": frame_error is not None,
        "reference_force_axis_cross_norm_m_s2": float(np.linalg.norm(np.cross(force_vector, axis))),
        "reference_force_projection_n": float(config["mass"] * np.dot(force_vector, axis)),
        "measured_pva_force_norm_n": measured_force, "reference_force_norm_n": force_n,
        "physical_force_bounds_n": [float(config["thrust_min"]), float(config["thrust_max"])],
        "reference_force_physical_lower_excess_n": max(float(config["thrust_min"]) - force_n, 0.0),
        "reference_force_physical_upper_excess_n": max(force_n - float(config["thrust_max"]), 0.0),
        "measured_pva_nominal_thrust_shortfall_n": max(nominal - measured_force, 0.0),
        "reference_nominal_thrust_shortfall_n": max(nominal - force_n, 0.0),
        "metadata_consistency": consistency,
        "scope": "Position/velocity are fixed in both modes. Only fixed_pva fixes measured acceleration; "
                 "ctbr_collective fixes the measured attitude and optimizes force magnitude. "
                 "Measured body rates are not fixed boundary states of this 10D model.",
    }


def unit_roots(poly):
    coefficients = np.trim_zeros(poly.coef, "b")
    if len(coefficients) <= 1:
        return []
    return [float(root.real) for root in Polynomial(coefficients).roots()
            if abs(root.imag) < 1e-8 and 0.0 < root.real < 1.0]


def polynomial_range(poly):
    times = np.asarray([0.0, 1.0, *unit_roots(poly.deriv())])
    values = poly(times)
    low, high = int(np.argmin(values)), int(np.argmax(values))
    return (float(values[low]), float(times[low])), (float(values[high]), float(times[high]))


class SavedCurve:
    """Evaluate serialized arrays without constructing PathCoverTrajectory."""

    def __init__(self, plan):
        trajectory = plan.trajectory
        self.coefficients = np.asarray(trajectory.coefficients, dtype=np.float64)
        self.durations = np.asarray(trajectory.durations, dtype=np.float64)
        self.boundaries = np.asarray(trajectory._boundaries, dtype=np.float64)
        self.yaw_times = np.asarray(trajectory._yaw_times, dtype=np.float64)
        self.yaw_values = np.asarray(trajectory._yaw_values, dtype=np.float64)
        rates = getattr(trajectory, "_yaw_rates", None)
        self.yaw_rates = None if rates is None else np.asarray(rates, dtype=np.float64)
        self.duration = float(self.boundaries[-1])
        arrays = [self.coefficients, self.durations, self.boundaries,
                  self.yaw_times, self.yaw_values]
        if self.yaw_rates is not None:
            arrays.append(self.yaw_rates)
        if not all(np.isfinite(array).all() for array in arrays):
            raise ValueError("Saved trajectory contains nonfinite data")
        if self.coefficients.shape != (len(self.durations), 3, 6):
            raise ValueError("Expected saved quintics [piece, axis, descending coefficient]")
        if not np.all(self.durations > 0):
            raise ValueError("Saved durations must be positive")
        if not np.array_equal(self.boundaries, np.r_[0.0, np.cumsum(self.durations)]):
            raise ValueError("Saved polynomial boundaries differ from saved durations")
        if (len(self.yaw_times) < 2 or self.yaw_times.shape != self.yaw_values.shape
                or np.any(np.diff(self.yaw_times) < 0)
                or self.yaw_times[0] != 0.0
                or abs(self.yaw_times[-1] - self.duration) > len(self.durations) * np.spacing(self.duration)):
            raise ValueError("Saved yaw knots do not cover the polynomial duration")
        if self.yaw_rates is not None and self.yaw_rates.shape != self.yaw_times.shape:
            raise ValueError("Saved yaw rates and knots have different shapes")
        if len(plan.segments) != len(self.durations):
            raise ValueError("Saved polynomial and segment counts differ")
        self.plan = plan

    def derivatives(self, times, *, side="left", pieces=None):
        times = np.asarray(times, dtype=np.float64).reshape(-1)
        if np.any(times < 0) or np.any(times > self.duration):
            raise ValueError("Curve evaluation requires the saved interval")
        if pieces is None:
            pieces = np.minimum(np.searchsorted(self.boundaries[1:], times, side=side),
                                len(self.durations) - 1)
        pieces = np.asarray(pieces)
        local = times - self.boundaries[pieces]
        coefficient = self.coefficients[pieces]
        output = []
        for order in range(4):
            value = np.zeros((len(times), 3))
            for column in range(6 - order):
                factor = 1
                for derivative in range(order):
                    factor *= 5 - column - derivative
                value = value * local[:, None] + coefficient[:, :, column] * factor
            output.append(value)
        return output

    def yaw(self, times, *, side="left"):
        times = np.asarray(times, dtype=np.float64).reshape(-1)
        index = np.clip(np.searchsorted(self.yaw_times[1:], times, side=side),
                        0, len(self.yaw_times) - 2)
        step = self.yaw_times[index + 1] - self.yaw_times[index]
        if self.yaw_rates is None:
            return (np.interp(times, self.yaw_times, self.yaw_values),
                    np.diff(self.yaw_values)[index] / step)
        fraction = (times - self.yaw_times[index]) / step
        s2, s3 = fraction**2, fraction**3
        y0, y1 = self.yaw_values[index], self.yaw_values[index + 1]
        d0, d1 = self.yaw_rates[index], self.yaw_rates[index + 1]
        yaw = ((2 * s3 - 3 * s2 + 1) * y0
               + (s3 - 2 * s2 + fraction) * step * d0
               + (-2 * s3 + 3 * s2) * y1 + (s3 - s2) * step * d1)
        rate = (6 * fraction * (1 - fraction) * (y1 - y0) / step
                + (3 * s2 - 4 * fraction + 1) * d0
                + (3 * s2 - 2 * fraction) * d1)
        # Deliberately no t >= duration rate clipping: this is the curve's
        # left limit. The deployed sampler's terminal behavior is reported
        # separately and must not erase a terminal curve peak.
        return yaw, rate

    def controls(self, times, mass, gravity, *, side="left", terminal_sampler=False):
        _, velocity, acceleration, jerk = self.derivatives(times, side=side)
        yaw, yaw_rate = self.yaw(times, side=side)
        if terminal_sampler:
            yaw_rate = np.where(np.asarray(times).reshape(-1) >= self.duration, 0.0, yaw_rate)
        force = acceleration + np.array([0.0, 0.0, gravity])
        norm = np.linalg.norm(force, axis=-1)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            z = force / norm[:, None]
            zdot = (jerk - z * np.sum(jerk * z, axis=-1)[:, None]) / norm[:, None]
            heading = np.stack((np.cos(yaw), np.sin(yaw), np.zeros_like(yaw)), axis=-1)
            heading_dot = yaw_rate[:, None] * np.stack(
                (-np.sin(yaw), np.cos(yaw), np.zeros_like(yaw)), axis=-1)
            cross = np.cross(z, heading)
            cross_norm = np.linalg.norm(cross, axis=-1)
            y = cross / cross_norm[:, None]
            cross_dot = np.cross(zdot, heading) + np.cross(z, heading_dot)
            ydot = (cross_dot - y * np.sum(y * cross_dot, axis=-1)[:, None]) / cross_norm[:, None]
            x = np.cross(y, z)
            rates = np.stack((-np.sum(y * zdot, axis=-1), np.sum(x * zdot, axis=-1),
                              -np.sum(x * ydot, axis=-1)), axis=-1)
            denominator = 1 + z[:, 2]
            term = zdot[:, 2] / denominator
            proxy = np.stack((-zdot[:, 1] + z[:, 1] * term,
                              zdot[:, 0] - z[:, 0] * term,
                              (z[:, 1] * zdot[:, 0] - z[:, 0] * zdot[:, 1]) / denominator), axis=-1)
        observables = np.column_stack((np.abs(rates), np.linalg.norm(rates, axis=-1),
                                       np.abs(yaw_rate), np.linalg.norm(proxy, axis=-1)))
        return {"observables": observables, "body_rates_rad_s": rates,
                "world_yaw_rate_rad_s": yaw_rate, "thrust_n": mass * norm,
                "tilt_rad": np.arccos(np.clip(z[:, 2], -1.0, 1.0)),
                "speed_m_s": np.linalg.norm(velocity, axis=-1)}

    def location(self, value, piece, unit_time):
        local = float(unit_time * self.durations[piece])
        relative = float(self.boundaries[piece] + local)
        return {"value": float(value), "piece": int(piece), "local_time_s": local,
                "relative_time_s": relative, "absolute_time_s": self.plan.start_time + relative}


def polynomial_extrema(curve, mass, gravity, per_face_margins=None):
    """Full original SFC, speed, force norm and tilt over every saved piece."""
    plan = curve.plan
    records = {name: [] for name in ("sfc_max_m", "speed_max_m_s", "thrust_min_n",
                                    "thrust_max_n", "tilt_max_rad")}
    velocities = [[[], []] for _ in range(3)]
    piece_records, normals = [], []
    margin_worst = []
    for piece, (coefficient, segment) in enumerate(zip(curve.coefficients, plan.segments, strict=True)):
        duration = curve.durations[piece]
        if duration != segment.duration:
            raise ValueError("Segment duration differs from saved polynomial")
        position = [Polynomial(axis[::-1] * duration ** np.arange(6)) for axis in coefficient]
        velocity = [poly.deriv() / duration for poly in position]
        acceleration = [poly.deriv() / duration for poly in velocity]
        force = acceleration.copy()
        force[2] = force[2] + gravity
        speed_squared = sum(poly * poly for poly in velocity)
        force_squared = sum(poly * poly for poly in force)
        _, speed_high = polynomial_range(speed_squared)
        force_low, force_high = polynomial_range(force_squared)
        speed = np.linalg.norm([poly(speed_high[1]) for poly in velocity])
        fmin = np.linalg.norm([poly(force_low[1]) for poly in force])
        fmax = np.linalg.norm([poly(force_high[1]) for poly in force])
        records["speed_max_m_s"].append(curve.location(speed, piece, speed_high[1]))
        records["thrust_min_n"].append(curve.location(mass * fmin, piece, force_low[1]))
        records["thrust_max_n"].append(curve.location(mass * fmax, piece, force_high[1]))
        tilt_stationary = 2 * force[2].deriv() * force_squared - force[2] * force_squared.deriv()
        times = np.array([0.0, 1.0, *unit_roots(tilt_stationary), force_low[1]])
        forces = np.stack([poly(times) for poly in force], axis=-1)
        force_norm = np.linalg.norm(forces, axis=-1)
        if np.any(force_norm == 0):
            raise ValueError("Saved reference has undefined zero-force attitude")
        tilt = np.arccos(np.clip(forces[:, 2] / force_norm, -1.0, 1.0))
        high = int(np.argmax(tilt))
        records["tilt_max_rad"].append(curve.location(tilt[high], piece, times[high]))
        for axis, poly in enumerate(velocity):
            low, high = polynomial_range(poly)
            velocities[axis][0].append(curve.location(low[0], piece, low[1]))
            velocities[axis][1].append(curve.location(high[0], piece, high[1]))
        faces, shifted_faces = [], []
        corridor_index = int(segment.corridor_index)
        corridor = np.asarray(plan.corridors[corridor_index])
        if corridor.ndim != 2 or corridor.shape[1] != 4 or not len(corridor):
            raise ValueError("Saved SFC must contain [nx, ny, nz, d] faces")
        for face_index, face in enumerate(corridor):
            scalar = sum(float(n) * poly for n, poly in zip(face[:3], position)) + face[3]  # noqa: B905
            maximum, when = polynomial_range(scalar)[1]
            normals.append(float(np.linalg.norm(face[:3])))
            face_record = {**curve.location(maximum, piece, when), "corridor": corridor_index,
                           "face": face_index, "plane": face.tolist()}
            faces.append(face_record)
            if per_face_margins is not None:
                margin = float(per_face_margins[corridor_index][face_index])
                shifted_faces.append({**face_record, "value": maximum + margin,
                                      "applied_margin_m": margin})
        worst = max(faces, key=lambda item: item["value"])
        records["sfc_max_m"].append(worst)
        piece_records.append({"piece": piece, "duration_s": float(duration),
                              "corridor": corridor_index, "worst_original_sfc": worst})
        if shifted_faces:
            margin_worst.append(max(shifted_faces, key=lambda item: item["value"]))
    result = {name: (min if name == "thrust_min_n" else max)(items, key=lambda item: item["value"])
              for name, items in records.items()}
    result["velocity_min_m_s"] = [min(items[0], key=lambda item: item["value"]) for items in velocities]
    result["velocity_max_m_s"] = [max(items[1], key=lambda item: item["value"]) for items in velocities]
    return {"method": "Normalized-time scalar polynomial derivative roots plus endpoints. "
                      "Tilt uses stationary roots of 2*fz_prime*force_squared-fz*force_squared_prime. "
                      "Floating-point roots, not interval-certified bounds.",
            "extrema": result, "piece_sfc_extrema": piece_records,
            "original_plane_normal_norm_range": [min(normals), max(normals)],
            "original_faces_evaluated": len(normals),
            "margin_adjusted_sfc_max": max(margin_worst, key=lambda item: item["value"]) if margin_worst else None}


def rate_extrema(curve, mass, gravity):
    """Dense saved-yaw controls plus local extrema, including endpoint limits."""
    uniform = np.linspace(0.0, curve.duration, max(2, int(np.ceil(curve.duration / 0.001)) + 1))
    knots = np.unique(np.r_[curve.boundaries, curve.yaw_times])
    # Native summation and NumPy cumsum can differ by one ULP. Keep the saved
    # Hermite arrays unchanged, as deployed; only restrict query times to the
    # polynomial's valid interval. Duplicate zero-width terminal knots carry no
    # interval. searchsorted(left) evaluates the actual preceding left limit.
    knots = knots[(knots >= 0.0) & (knots <= curve.duration)]
    grid = np.unique(np.r_[uniform, knots])
    values = curve.controls(grid, mass, gravity)["observables"]
    right_times = knots[(knots > 0.0) & (knots < curve.duration)]
    right_values = curve.controls(right_times, mass, gravity, side="right")["observables"]
    if not np.isfinite(values).all() or not np.isfinite(right_values).all():
        raise ValueError("Saved reference has nonfinite flatness or zero-yaw proxy values")
    extrema = {}
    for axis, name in enumerate(RATE_NAMES):
        column = values[:, axis]
        best = int(np.argmax(column))
        best_value, best_time, best_side = float(column[best]), float(grid[best]), "left"
        if len(right_times):
            right_best = int(np.argmax(right_values[:, axis]))
            if right_values[right_best, axis] > best_value:
                best_value = float(right_values[right_best, axis])
                best_time, best_side = float(right_times[right_best]), "right"
        local = np.flatnonzero((column[1:-1] >= column[:-2]) & (column[1:-1] >= column[2:])) + 1
        local = local[np.argsort(column[local])[-8:]]
        trials = []
        for i in local:
            left, right = float(grid[i - 1]), float(grid[i + 1])
            splits = np.r_[left, knots[(knots > left) & (knots < right)], right]
            for lower, upper in zip(splits[:-1], splits[1:]):  # noqa: B905
                if upper - lower < 1e-11:
                    continue
                result = minimize_scalar(
                    lambda time: -float(curve.controls([time], mass, gravity)["observables"][0, axis]),  # noqa: B023
                    bounds=(float(lower), float(upper)), method="bounded",
                    options={"xatol": 1e-11})
                trials.append({"bounds_s": [float(lower), float(upper)], "success": bool(result.success)})
                if -result.fun > best_value:
                    best_value, best_time, best_side = float(-result.fun), float(result.x), "interior"
        extrema[name] = {"value": best_value, "relative_time_s": best_time,
                         "absolute_time_s": curve.plan.start_time + best_time, "side": best_side,
                         "grid_maximum": float(max(column.max(), right_values[:, axis].max()
                                                   if len(right_values) else -np.inf)),
                         "local_refinements": len(trials),
                         "all_local_refinements_succeeded": all(item["success"] for item in trials)}
    def terminal(value):
        return {key: array[0].tolist() for key, array in value.items() if key != "observables"}
    return {"method": "Full <=1 ms grid, every stored yaw/polynomial knot with both one-sided values, "
                      "and bounded refinement of the eight strongest sampled interior local maxima per observable. "
                      "Brackets are split at knots. Numerical localization, not certified global bounds.",
            "grid_samples": len(grid), "right_limit_samples": len(right_times),
            "maximum_grid_gap_s": float(np.diff(grid).max()), "extrema": extrema,
            "saved_yaw_metadata": {
                "stored_final_knot_s": float(curve.yaw_times[-1]),
                "polynomial_duration_s": curve.duration,
                "final_knot_minus_duration_s": float(curve.yaw_times[-1] - curve.duration),
                "zero_width_interval_indices": np.flatnonzero(np.diff(curve.yaw_times) == 0).tolist(),
                "semantics": "Original saved arrays and searchsorted interval selection retained; no yaw reconstruction or node editing."},
            "terminal_curve_left_limit": terminal(curve.controls([curve.duration], mass, gravity)),
            "deployed_sample_at_duration": terminal(
                curve.controls([curve.duration], mass, gravity, terminal_sampler=True)),
            "strictly_after_expiry_hold": {"velocity_m_s": [0.0] * 3, "acceleration_m_s2": [0.0] * 3,
                "jerk_m_s3": [0.0] * 3, "body_rates_rad_s": [0.0] * 3,
                "world_yaw_rate_rad_s": 0.0, "thrust_n": mass * abs(gravity)},
            "endpoint_semantics": "Curve extrema include the pre-expiry terminal left limit. "
                "PathCoverTrajectory.sample sets yaw_rate=0 at t>=duration but keeps polynomial derivatives at equality; "
                "it sets all derivatives to zero only strictly after duration. These three semantics are separate."}


class ArchiveIntegrityError(ValueError):
    pass


def clean(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    return value


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def event_key(record):
    stamp, sequence = record.get("stamp"), record.get("plan_sequence_id")
    if (isinstance(stamp, bool) or not isinstance(stamp, numbers.Real)
            or not np.isfinite(stamp) or isinstance(sequence, bool)
            or not isinstance(sequence, numbers.Integral) or sequence <= 0):
        raise ArchiveIntegrityError("Every solve/archive entry needs finite stamp and positive integer plan_sequence_id")
    return float(stamp), int(sequence)


def select_records(archive, evaluator_rows, metadata):
    """Exact (stamp, sequence) join. Missing and duplicate rows are not dropped."""
    if archive.get("schema_version") != 1:
        raise ArchiveIntegrityError("Only the recorded reference archive schema_version=1 is supported")
    if metadata.get("parameters", {}).get("frontend") != "pathcover":
        raise ArchiveIntegrityError("Reference archive analysis requires a PathCover run")
    config = archive.get("frontend_config")
    if clean(config) != clean(metadata.get("frontend_config")):
        raise ArchiveIntegrityError("Archive frontend_config differs from this run's metadata")
    expected = [row for row in evaluator_rows if row.get("event") == "solve"]
    records = archive.get("records")
    if not isinstance(records, list):
        raise ArchiveIntegrityError("Archive records must be a list")
    def indexed(rows, name):
        output, sequences = {}, set()
        for row in rows:
            key = event_key(row)
            if key in output or key[1] in sequences:
                raise ArchiveIntegrityError(f"Duplicate {name} stamp/sequence or sequence: {key}")
            output[key] = row
            sequences.add(key[1])
        return output
    expected_index = indexed(expected, "evaluator solve")
    archive_index = indexed(records, "archive")
    missing = sorted(set(expected_index) - set(archive_index))
    if missing:
        raise ArchiveIntegrityError(f"Archive misses {len(missing)} evaluated solve(s): {missing[:8]}")
    selected = []
    for row in expected:
        record = archive_index[event_key(row)]
        if record.get("frontend_plan") is None:
            raise ArchiveIntegrityError(f"Evaluated solve has no saved CPU plan: {event_key(row)}")
        if ("state_time" in row and record.get("state_time") != row["state_time"]):
            raise ArchiveIntegrityError(f"Saved and evaluated state_time differ: {event_key(row)}")
        selected.append((record, row))
    extra = [record for record in records if event_key(record) not in expected_index]
    last_stamp = max((key[0] for key in expected_index), default=None)
    last_sequence = max((key[1] for key in expected_index), default=None)
    return selected, {
        "schema_version": 1, "evaluator_solve_count": len(expected),
        "archive_record_count": len(records), "matched_record_count": len(selected),
        "missing_count": 0, "duplicate_count": 0, "complete_for_evaluator": True,
        "selection": "Exact numeric (stamp, plan_sequence_id) membership in evaluator solve rows; no time epsilon, nearest match or duration-only cutoff.",
        "excluded_archive_records": [
            {"stamp": float(record["stamp"]), "plan_sequence_id": int(record["plan_sequence_id"]),
             "after_last_evaluator_solve": last_stamp is not None and record["stamp"] > last_stamp,
             "after_last_evaluator_sequence": last_sequence is not None and record["plan_sequence_id"] > last_sequence,
             "reason": "No matching solve received by evaluator; excluded from all reference aggregates"}
            for record in extra],
        "excluded_count": len(extra),
        "failed_attempts_without_returned_plan": sum(row.get("event") == "failure" for row in evaluator_rows),
        "control_events_are_not_new_plans": sum(row.get("event") == "control" for row in evaluator_rows),
    }


def margin_data(plan, row):
    recorded = {key: clean(value) for key, value in plan.metrics.items() if key.startswith("reference_margin")}
    for key, value in row.items():
        if key.startswith("reference_margin"):
            if key in recorded and clean(value) != recorded[key]:
                raise ArchiveIntegrityError(f"Plan/evaluator margin metadata differ: {key}")
            recorded[key] = clean(value)
    per_face = recorded.get("reference_margin_applied_per_face_m")
    if per_face is not None:
        if len(per_face) != len(plan.corridors) or any(
                len(values) != len(faces) for values, faces in zip(per_face, plan.corridors, strict=True)):
            raise ArchiveIntegrityError("Per-face margin shape does not match original corridors")
        representation = "Recorded per-face offsets. Per-corridor applied values are minima, not uniform shifts."
    elif (recorded.get("reference_margin_applied_per_corridor_m") is not None
          and "reference_margin_cap_per_corridor_m" not in recorded):
        legacy = recorded["reference_margin_applied_per_corridor_m"]
        if len(legacy) != len(plan.corridors):
            raise ArchiveIntegrityError("Historical uniform margin count differs from corridors")
        per_face = [[float(margin)] * len(faces) for margin, faces in zip(legacy, plan.corridors, strict=True)]
        representation = "Historical scalar-per-corridor uniform offsets, expanded only for this offline comparison."
    else:
        representation = "Per-face offsets unavailable; no margin-adjusted extrema inferred from corridor minima."
    if per_face is not None and not all(np.isfinite(values).all() for values in per_face):
        raise ArchiveIntegrityError("Nonfinite saved margins")
    return {"recorded": recorded, "interpretation": representation}, per_face


def thresholds(config, model, row, plan):
    measured = {**plan.metrics, **row}
    rate_limit = float(config["body_rate_max"])
    margin = float(config.get("reference_body_rate_margin", 0.0))
    nominal_thrust = config.get("nominal_thrust_min")
    nominal_thrust = float(config["thrust_min"] if nominal_thrust is None else nominal_thrust)
    return {
        "reference": {
            "mass_kg": float(config["mass"]), "gravity_m_s2": float(config["gravity"]),
            "thrust_n": [float(config["thrust_min"]), float(config["thrust_max"])],
            "nominal_thrust_min_n": nominal_thrust,
            "tilt_rad": float(config["tilt_max"]),
            "zero_yaw_proxy_norm_rad_s": rate_limit,
            "optimizer_zero_yaw_proxy_target_rad_s": rate_limit - margin,
            "reference_body_rate_margin_rad_s": margin,
            "body_yaw_rate_max_rad_s": float(config["body_yaw_rate_max"]),
            "nominal_speed_m_s": float(config["max_velocity"] if config.get("nominal_velocity") is None else config["nominal_velocity"]),
            "physical_speed_norm_m_s": float(config["max_velocity"]),
            "recorded_state_conditioned_speed_threshold_m_s": measured.get("speed_acceptance_limit_m_s"),
        },
        "ctbr": {
            "mass_kg": float(model["mass"]), "gravity_m_s2": float(model["gravity"]),
            "thrust_n": [float(model["mass"] * value) for value in model["specific_thrust_limits"]],
            "body_rate_components_rad_s": list(model["body_rate_limits"]),
            "velocity_lower_m_s": list(model["velocity_lower"]),
            "velocity_upper_m_s": list(model["velocity_upper"]), "hard_tilt_bound": None,
        },
        "diagnostic_allowance": {
            "original_sfc_m": 1e-4, "reference_thrust_n": 1e-3,
            "reference_tilt_rad": 1e-3, "zero_yaw_proxy_norm_rad_s": 1e-3,
            "ctbr_axis_comparison_allowance_rad_s": 1e-3,
            "scope": "Raw bound excesses are also reported. The optional 1e-3 axis comparison is an offline diagnostic, not a modified MPC bound or solver tolerance.",
        },
        "mass_and_gravity_match": config["mass"] == model["mass"] and config["gravity"] == model["gravity"],
        "scope": "Reference flatness controls use the recorded frontend mass/gravity. CTBR limits come from this run's solver.model, never standalone frontend defaults. This comparison does not solve or certify CTBR discrete dynamics.",
    }


def quadrature_diagnostics(metrics, bounds):
    reference = bounds["reference"]
    tests = (
        ("sfc", "optimizer_sfc_max_m", "upper", 1e-4),
        ("speed_norm", "optimizer_speed_max_m_s", "upper", reference["recorded_state_conditioned_speed_threshold_m_s"]),
        ("thrust_low", "optimizer_thrust_min_n", "lower", reference["thrust_n"][0] - 1e-3),
        ("thrust_high", "optimizer_thrust_max_n", "upper", reference["thrust_n"][1] + 1e-3),
        ("tilt", "optimizer_tilt_max_rad", "upper", reference["tilt_rad"] + 1e-3),
        ("zero_yaw_proxy_norm", "optimizer_zero_yaw_body_rate_max_rad_s", "upper", reference["zero_yaw_proxy_norm_rad_s"] + 1e-3),
    )
    violations, unknown = [], []
    for category, key, direction, bound in tests:
        value = metrics.get(key)
        if value is None or bound is None:
            unknown.append(category)
        elif (value > bound if direction == "upper" else value < bound):
            violations.append(category)
    flag = metrics.get("optimizer_quadrature_feasible")
    minimum = metrics.get("optimizer_thrust_min_n")
    return {"recorded_feasible": flag, "violations": violations, "unclassified_missing_fields": unknown,
            "recorded_flag_matches_reconstructed_predicate": flag == (not violations) if isinstance(flag, bool) and not unknown else None,
            "recorded_metrics": {key: clean(value) for key, value in metrics.items() if key.startswith("optimizer_")},
            "nominal_thrust_shortfall_n": None if minimum is None else max(reference["nominal_thrust_min_n"] - minimum, 0.0),
            "semantics": "Nominal thrust shortfall is separate from the physical predicate. Zero-yaw norm proxy is separate from saved-yaw body-axis controls; a false flag does not by itself establish an actual-axis, MPC or measured execution failure."}


def quality_of_record(record, row, config, model):
    plan = record["frontend_plan"]
    curve = SavedCurve(plan)
    margin, per_face = margin_data(plan, row)
    limits = thresholds(config, model, row, plan)
    mass, gravity = float(config["mass"]), float(config["gravity"])
    poly = polynomial_extrema(curve, mass, gravity, per_face)
    rates = rate_extrema(curve, mass, gravity)
    extreme = poly["extrema"]
    observed_rates = np.asarray([rates["extrema"][key]["value"] for key in ("abs_body_p", "abs_body_q", "abs_body_r")])
    ctbr = limits["ctbr"]
    velocity_low = np.asarray([item["value"] for item in extreme["velocity_min_m_s"]])
    velocity_high = np.asarray([item["value"] for item in extreme["velocity_max_m_s"]])
    axis_excess = observed_rates - np.asarray(ctbr["body_rate_components_rad_s"])
    comparisons = {
        "original_sfc_signed_max_m": extreme["sfc_max_m"]["value"],
        "original_sfc_positive_violation_m": max(0.0, extreme["sfc_max_m"]["value"]),
        "original_sfc_above_1e_4_m": extreme["sfc_max_m"]["value"] > 1e-4,
        "saved_yaw_actual_axis_signed_excess_rad_s": axis_excess.tolist(),
        "saved_yaw_actual_axis_positive_excess_rad_s": np.maximum(axis_excess, 0).tolist(),
        "saved_yaw_actual_axis_above_ctbr_bounds": (axis_excess > 0).tolist(),
        "saved_yaw_actual_axis_above_ctbr_plus_1e_3": (axis_excess > 1e-3).tolist(),
        "reference_thrust_lower_positive_excess_n": max(0.0, config["thrust_min"] - extreme["thrust_min_n"]["value"]),
        "reference_nominal_thrust_shortfall_n": max(0.0, limits["reference"]["nominal_thrust_min_n"] - extreme["thrust_min_n"]["value"]),
        "reference_thrust_upper_positive_excess_n": max(0.0, extreme["thrust_max_n"]["value"] - config["thrust_max"]),
        "ctbr_thrust_lower_positive_excess_n": max(0.0, ctbr["thrust_n"][0] - extreme["thrust_min_n"]["value"]),
        "ctbr_thrust_upper_positive_excess_n": max(0.0, extreme["thrust_max_n"]["value"] - ctbr["thrust_n"][1]),
        "reference_tilt_positive_excess_rad": max(0.0, extreme["tilt_max_rad"]["value"] - config["tilt_max"]),
        "reference_physical_speed_positive_excess_m_s": max(0.0, extreme["speed_max_m_s"]["value"] - config["max_velocity"]),
        "ctbr_velocity_lower_positive_excess_m_s": np.maximum(np.asarray(ctbr["velocity_lower_m_s"]) - velocity_low, 0).tolist(),
        "ctbr_velocity_upper_positive_excess_m_s": np.maximum(velocity_high - np.asarray(ctbr["velocity_upper_m_s"]), 0).tolist(),
    }
    initial = curve.derivatives([0.0])
    terminal = curve.derivatives([curve.duration])
    measured_initial_pva = np.column_stack((record["state"][:3], record["state"][7:10], record["acceleration"]))
    actual_initial_pva = np.column_stack([value[0] for value in initial[:3]])
    boundary_metadata = plan.metrics.get("initial_boundary")
    if "initial_boundary" in row:
        if boundary_metadata is not None and clean(boundary_metadata) != clean(row["initial_boundary"]):
            raise ArchiveIntegrityError("Plan/evaluator initial_boundary metadata differ")
        boundary_metadata = row["initial_boundary"]
    boundary = initial_boundary_contract(curve, record["state"], record["acceleration"], config, boundary_metadata)
    expected_terminal_pva = np.column_stack((plan.local_goal, np.zeros(3), np.zeros(3)))
    actual_terminal_pva = np.column_stack([value[0] for value in terminal[:3]])
    node_times = record["state_time"] - plan.start_time + np.arange(model["cost_nodes"]) * model["dt"]
    node_position = curve.derivatives(np.clip(node_times, 0, curve.duration))[0]
    node_indices = np.minimum(np.searchsorted(curve.boundaries[1:], np.clip(node_times, 0, curve.duration), side="left"),
                              len(curve.durations) - 1)
    nodes = []
    for relative, position, piece in zip(node_times, node_position, node_indices, strict=True):
        ci = int(plan.segments[int(piece)].corridor_index)
        faces = np.asarray(plan.corridors[ci])
        maximum = float(np.max(faces[:, :3] @ position + faces[:, 3]))
        nodes.append({"relative_time_s": float(relative), "piece": int(piece), "corridor": ci,
                      "original_sfc_signed_max_m": maximum})
    metrics = {**plan.metrics, **row}
    return clean({
        "stamp": record["stamp"], "state_time": record["state_time"],
        "plan_sequence_id": record["plan_sequence_id"],
        "start_time_s": plan.start_time, "valid_until_s": plan.valid_until,
        "duration_s": curve.duration, "valid_until_minus_polynomial_end_s": plan.valid_until - (plan.start_time + curve.duration),
        "task_goal": record["goal"], "local_goal": plan.local_goal,
        "state": record["state"], "measured_body_rates_rad_s": record["body_rates"],
        "initial_pva": measured_initial_pva,
        "measured_initial_pva": measured_initial_pva,
        "reference_initial_pva": actual_initial_pva,
        "initial_pva_difference_by_column": np.max(np.abs(actual_initial_pva - measured_initial_pva), axis=0),
        "initial_pva_error_by_column": (np.max(np.abs(actual_initial_pva - measured_initial_pva), axis=0)
                                        if boundary["mode"] == "fixed_pva" else None),
        "initial_boundary": boundary,
        "terminal_pva_error_by_column": np.max(np.abs(actual_terminal_pva - expected_terminal_pva), axis=0),
        "coefficient_sha256": hashlib.sha256(curve.coefficients.tobytes()).hexdigest(),
        "saved_yaw_sha256": hashlib.sha256(curve.yaw_times.tobytes() + curve.yaw_values.tobytes()
                                           + (b"" if curve.yaw_rates is None else curve.yaw_rates.tobytes())).hexdigest(),
        "bounds": limits, "reference_margins": margin,
        "full_polynomial": poly, "actual_saved_yaw": rates,
        "comparisons": comparisons, "quadrature": quadrature_diagnostics(metrics, limits),
        "current_reference_nodes": nodes,
        "logged_current_reference_sfc_violation_m": row.get("reference_sfc_violation_m"),
        "logged_predicted_sfc_violation_m": row.get("predicted_sfc_violation_m"),
        "logged_mpc_solved": row.get("qp_ok"),
        "physical_scope": "Whole returned reference, including future parts that may be replanned before execution. These extrema are neither actual vehicle measurements nor newly solved MPC trajectories.",
    })


class PlanCache:
    """Independent per-plan results, reusable only for identical inputs/code."""

    def __init__(self, directory, provenance):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.provenance = provenance
        manifest = self.directory / "manifest.json"
        content = {"schema_version": 1, "provenance": provenance,
                   "scope": "Per-plan checkpoints only. No complete aggregate until strict coverage and all plans pass analysis."}
        if manifest.exists():
            if json.loads(manifest.read_text()) != content:
                raise ArchiveIntegrityError("Reference cache inputs or analysis source changed; use a new cache directory")
        else:
            self.write(manifest, content)
        self.reused = 0
        self.computed = 0

    @staticmethod
    def write(path, content):
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)

    def get(self, record):
        path = self.directory / f"plan-{record['plan_sequence_id']:06d}.json"
        if not path.exists():
            return None
        result = json.loads(path.read_text())
        if event_key(result) != event_key(record):
            raise ArchiveIntegrityError("Cached reference identity differs from the selected archive record")
        self.reused += 1
        return result

    def put(self, record, result):
        self.write(self.directory / f"plan-{record['plan_sequence_id']:06d}.json", result)
        self.computed += 1

    def progress(self, completed, total):
        value = {"completed_plans": completed, "total_selected_plans": total,
                 "computed_this_invocation": self.computed, "reused_this_invocation": self.reused,
                 "all_selected_plans_analyzed": completed == total,
                 "aggregate_available": False}
        self.write(self.directory / "progress.json", value)
        print(json.dumps(value, separators=(",", ":")), flush=True)


def motion_seed_adjustment_summary(selected):
    field = "initial_motion_seed_adjusted"
    records = []
    for record, _ in selected:
        metrics = record["frontend_plan"].metrics
        if field not in metrics:
            continue
        value = metrics[field]
        if not isinstance(value, (bool, np.bool_)):
            raise ArchiveIntegrityError(f"Non-boolean {field} in returned plan {event_key(record)}")
        records.append({"stamp": record["stamp"], "plan_sequence_id": record["plan_sequence_id"],
                        "adjusted": bool(value)})
    return {
        "scope": "Evaluator-matched returned plans only. Missing flags remain unknown. No geometric thrust is inferred from MPC controls.",
        "returned_plan_count": len(selected),
        "field_present_plan_count": len(records),
        "field_missing_plan_count": len(selected) - len(records),
        "adjusted_true_count": sum(record["adjusted"] for record in records),
        "records_with_field": records,
    }


def analyze_archive(archive, evaluator_rows, metadata, *, cache=None):
    selected, coverage = select_records(archive, evaluator_rows, metadata)
    model, config = metadata["solver"]["model"], archive["frontend_config"]
    plans = []
    if cache is not None:
        print(json.dumps({"strict_archive_coverage": coverage}, separators=(",", ":")), flush=True)
    for index, (record, row) in enumerate(selected, start=1):
        result = cache.get(record) if cache is not None else None
        if result is None:
            result = quality_of_record(record, row, config, model)
            if cache is not None:
                cache.put(record, result)
        plans.append(result)
        if cache is not None and (index % 100 == 0 or index == len(selected)):
            cache.progress(index, len(selected))
    false = [plan for plan in plans if plan["quadrature"]["recorded_feasible"] is False]
    continuous_sfc = [plan for plan in plans if plan["comparisons"]["original_sfc_above_1e_4_m"]]
    axis_excess = [plan for plan in plans if any(plan["comparisons"]["saved_yaw_actual_axis_above_ctbr_plus_1e_3"])]
    def worst(selector):
        return max(plans, key=selector) if plans else None
    def identity(plan, value):
        return None if plan is None else {"stamp": plan["stamp"], "plan_sequence_id": plan["plan_sequence_id"], "value": value(plan)}
    sfc_getter = lambda plan: plan["comparisons"]["original_sfc_signed_max_m"]  # noqa: E731
    report = {
        "scope": "CPU-only evaluation of existing saved polynomials and saved yaw. No mapping, search, RISP, optimizer, native yaw generation, solver or GPU work.",
        "coverage": coverage, "frontend_config": clean(config), "ctbr_model": clean(model),
        "summary": {
            "analyzed_plans": len(plans), "recorded_quadrature_false": len(false),
            "initial_boundary_mode_counts": dict(Counter(plan["initial_boundary"]["mode"] for plan in plans)),
            "initial_boundary_missing_metadata_count": sum(not plan["initial_boundary"]["metadata_present"] for plan in plans),
            "maximum_initial_reference_measured_acceleration_difference_m_s2": max(
                (float(plan["initial_pva_difference_by_column"][2]) for plan in plans), default=None),
            "maximum_ctbr_initial_attitude_error_rad": max(
                (plan["initial_boundary"]["fixed_attitude_rotation_error_rad"] for plan in plans
                 if plan["initial_boundary"]["fixed_attitude_rotation_error_rad"] is not None), default=None),
            "ctbr_initial_attitude_undefined_count": sum(
                plan["initial_boundary"]["mode"] == "ctbr_collective"
                and not plan["initial_boundary"]["reference_upright_frame_defined"] for plan in plans),
            "reference_nominal_thrust_shortfall_plans": sum(
                plan["comparisons"]["reference_nominal_thrust_shortfall_n"] > 0.0 for plan in plans),
            "initial_motion_seed_adjustment": motion_seed_adjustment_summary(selected),
            "recorded_quadrature_unknown": sum(plan["quadrature"]["recorded_feasible"] is None for plan in plans),
            "quadrature_category_counts": dict(Counter(category for plan in plans for category in plan["quadrature"]["violations"])),
            "quadrature_predicate_mismatches": sum(plan["quadrature"]["recorded_flag_matches_reconstructed_predicate"] is False for plan in plans),
            "full_curve_original_sfc_above_1e_4": len(continuous_sfc),
            "actual_saved_yaw_any_axis_above_ctbr_plus_1e_3": len(axis_excess),
            "quadrature_false_without_actual_axis_excess_plus_1e_3": sum(
                not any(plan["comparisons"]["saved_yaw_actual_axis_above_ctbr_plus_1e_3"]) for plan in false),
            "worst_full_curve_original_sfc_m": identity(worst(sfc_getter), sfc_getter),
            "worst_actual_axis_rad_s": [
                identity(worst(lambda plan: plan["actual_saved_yaw"]["extrema"][axis]["value"]),  # noqa: B023
                         lambda plan: plan["actual_saved_yaw"]["extrema"][axis]["value"])  # noqa: B023
                for axis in ("abs_body_p", "abs_body_q", "abs_body_r")],
        },
        "limits": [
            "SFC, speed, thrust and tilt extrema use numerical polynomial roots, not interval arithmetic.",
            "Saved-yaw rate peaks use <=1 ms sampling, every knot and endpoint side, and local refinement; they are not certified global bounds.",
            "No captured false proxy flag is relabeled as actual body-axis failure without evaluating the saved-yaw frame.",
            "Reference soft quality, MPC hard-constraint diagnostics and measured execution metrics remain separate.",
            "Only evaluator-matched plans enter aggregates. Post-evaluator shutdown records remain listed under coverage.",
        ],
        "plans": plans,
    }
    return report


def quality_failure_reasons(plan):
    """Apply the established complete-reference allowances to one result.

    A nominal thrust shortfall and a zero-yaw quadrature proxy are diagnostics,
    not substitutes for the physical force and saved-yaw body-axis checks.
    """
    c = plan["comparisons"]
    b = plan["initial_boundary"]
    e = plan["full_polynomial"]["extrema"]
    threshold = plan["bounds"]["reference"]["recorded_state_conditioned_speed_threshold_m_s"]
    if threshold is None:
        raise ArchiveIntegrityError("Saved physical speed threshold missing")
    reasons = [name for name, failed in (
        ("original_sfc", c["original_sfc_above_1e_4_m"]),
        ("actual_body_axes", any(c["saved_yaw_actual_axis_above_ctbr_plus_1e_3"])),
        ("physical_thrust_low", c["reference_thrust_lower_positive_excess_n"] > 1e-3),
        ("physical_thrust_high", c["reference_thrust_upper_positive_excess_n"] > 1e-3),
        ("reference_tilt", c["reference_tilt_positive_excess_rad"] > 1e-3),
        ("physical_speed_norm", e["speed_max_m_s"]["value"] > threshold),
        ("ctbr_thrust_low", c["ctbr_thrust_lower_positive_excess_n"] > 1e-3),
        ("ctbr_thrust_high", c["ctbr_thrust_upper_positive_excess_n"] > 1e-3),
        ("ctbr_velocity_components", max(c["ctbr_velocity_lower_positive_excess_m_s"]
                                        + c["ctbr_velocity_upper_positive_excess_m_s"]) > 1e-6),
        ("initial_position", b["fixed_position_max_abs_error_m"] > 1e-8),
        ("initial_velocity", b["fixed_velocity_max_abs_error_m_s"] > 1e-8),
        ("terminal_pva", max(plan["terminal_pva_error_by_column"]) > 1e-8),
    ) if failed]
    if b["mode"] == "ctbr_collective":
        if b["fixed_attitude_rotation_error_rad"] is None or b["fixed_attitude_rotation_error_rad"] > 1e-8:
            reasons.append("initial_attitude")
    elif b["fixed_acceleration_max_abs_error_m_s2"] > 1e-8:
        reasons.append("initial_acceleration")
    return reasons


def summarize_quality(report):
    """Compact the complete results without reevaluating any saved curve."""
    plans, coverage = report["plans"], report["coverage"]
    if not coverage["complete_for_evaluator"] or not (
        len(plans) == coverage["evaluator_solve_count"] == coverage["matched_record_count"]
    ):
        raise ArchiveIntegrityError("Complete-reference summary requires exact coverage")

    def identity(plan):
        return {"plan_sequence_id": plan["plan_sequence_id"], "stamp": plan["stamp"]}

    def peak(group, key, minimum=False):
        if not plans:
            return None
        plan = (min if minimum else max)(plans, key=lambda p: p[group]["extrema"][key]["value"])
        return {**identity(plan), **plan[group]["extrema"][key]}

    extrema = {key: peak("full_polynomial", key, key == "thrust_min_n") for key in
               ("sfc_max_m", "speed_max_m_s", "thrust_min_n", "thrust_max_n", "tilt_max_rad")}
    rates = {key: peak("actual_saved_yaw", key) for key in RATE_NAMES}
    failures = []
    for plan in plans:
        reasons = quality_failure_reasons(plan)
        if reasons:
            failures.append({**identity(plan), "categories": reasons, "comparisons": plan["comparisons"],
                             "initial_boundary": plan["initial_boundary"],
                             "terminal_pva_error_by_column": plan["terminal_pva_error_by_column"]})
    boundaries = {key: max((p["initial_boundary"][key] for p in plans), default=None) for key in
                  ("fixed_position_max_abs_error_m", "fixed_velocity_max_abs_error_m_s",
                   "reference_force_physical_lower_excess_n", "reference_force_physical_upper_excess_n")}
    boundaries["fixed_attitude_max_error_rad"] = max(
        (p["initial_boundary"]["fixed_attitude_rotation_error_rad"] for p in plans
         if p["initial_boundary"]["fixed_attitude_rotation_error_rad"] is not None), default=None)
    boundaries["terminal_pva_max_abs_error_by_column"] = [
        max((p["terminal_pva_error_by_column"][i] for p in plans), default=None) for i in range(3)]
    residuals = {key: max((p["comparisons"][key] for p in plans), default=None) for key in
                 ("original_sfc_positive_violation_m", "reference_physical_speed_positive_excess_m_s",
                  "reference_thrust_lower_positive_excess_n", "reference_thrust_upper_positive_excess_n",
                  "ctbr_thrust_lower_positive_excess_n", "ctbr_thrust_upper_positive_excess_n",
                  "reference_tilt_positive_excess_rad", "reference_nominal_thrust_shortfall_n")}
    for key in ("ctbr_velocity_lower_positive_excess_m_s", "ctbr_velocity_upper_positive_excess_m_s",
                "saved_yaw_actual_axis_signed_excess_rad_s", "saved_yaw_actual_axis_positive_excess_rad_s"):
        residuals[key] = [max((p["comparisons"][key][i] for p in plans), default=None) for i in range(3)]
    quadrature = [{**identity(p), "quadrature": p["quadrature"],
                   "actual_axes_rad_s": [p["actual_saved_yaw"]["extrema"][key]["value"]
                                         for key in ("abs_body_p", "abs_body_q", "abs_body_r")],
                   "actual_axis_above_ctbr_plus_1e_3": p["comparisons"]["saved_yaw_actual_axis_above_ctbr_plus_1e_3"]}
                  for p in plans if p["quadrature"]["recorded_feasible"] is False]
    return {
        "reference_quality_passed": not failures if plans else None,
        "empty_archive_semantics": "No returned plans means no reference-quality conclusion, not a successful mission.",
        "raw_polynomial_extrema": extrema, "actual_saved_yaw_extrema": rates,
        "raw_bound_residuals": residuals, "boundary_residuals": boundaries,
        "failure_count": len(failures), "failures": failures,
        "failure_category_counts": dict(Counter(reason for item in failures for reason in item["categories"])),
        "quadrature_false_records": quadrature,
        "bounds": plans[0]["bounds"] if plans else None,
        "comparison_allowances": {
            "original_sfc_m": 1e-4, "actual_body_axis_rad_s": 1e-3,
            "physical_thrust_n": 1e-3, "reference_tilt_rad": 1e-3,
            "ctbr_velocity_m_s": 1e-6, "initial_fixed_and_terminal_boundary": 1e-8},
        "scope": "All evaluator-matched saved plans. Excluded records remain in coverage. "
                 "Free initial collective fixes position/velocity/attitude, not measured acceleration. "
                 "Failures to return a plan are retained separately and cannot pass a curve-quality test.",
    }


def planning_attempts(evaluator_rows):
    """Keep failed planning attempts and MPC outcomes separate from plan quality."""
    solves = [row for row in evaluator_rows if row.get("event") == "solve"]
    failures = [clean(row) for row in evaluator_rows if row.get("event") == "failure"]
    return {
        "attempt_count": len(solves) + len(failures),
        "returned_plan_count": len(solves),
        "returned_mpc_solved_count": sum(row.get("qp_ok") is True for row in solves),
        "returned_mpc_unsolved_count": sum(row.get("qp_ok") is False for row in solves),
        "returned_mpc_status_unknown_count": sum(not isinstance(row.get("qp_ok"), bool) for row in solves),
        "failed_attempt_count": len(failures), "failed_attempts": failures,
        "control_event_count": sum(row.get("event") == "control" for row in evaluator_rows),
        "scope": "Only evaluator-received solve/failure events are planning attempts. "
                 "Failure records, including their reported latency/error, are preserved verbatim. "
                 "Control publications do not add plans or turn an unsuccessful attempt into a solved MPC.",
    }
