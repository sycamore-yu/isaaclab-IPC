"""Cost-independent, local CTBR braking witnesses for reference progression.

The witness uses the canonical discrete prediction model, not Newton physics.
Virtual-speed witnesses additionally fit their whole horizontal excursion in
the current-height inscribed disc of the same unchanged polytope. This finite,
single-trajectory reference policy is not a certificate for the measured state,
the returned MPC schedule, arbitrary maneuvers, or future unseen obstacles.
"""

from dataclasses import dataclass, replace
from enum import IntEnum

import numpy as np

REFERENCE_GOVERNOR_VERSION = 2
REFERENCE_DIAGNOSTIC_FIELDS = (
    "version",
    "applied",
    "reason_code",
    "actual_backup_certified",
    "nominal_backup_certified",
    "selected_backup_certified",
    "reference_scale",
    "nominal_speed",
    "selected_speed",
    "actual_backup_steps",
    "selected_backup_steps",
    "containing_corridor",
)


class ReferenceReason(IntEnum):
    IDENTITY_CERTIFIED = 0
    ADAPTED_LOCAL_BACKUP = 1
    INVALID_CONTEXT = 2
    NO_FORWARD_PATH = 3
    NO_CONTAINING_CORRIDOR = 4
    ACTUAL_BACKUP_NOT_CERTIFIED = 5
    NO_VIRTUAL_BACKUP = 6
    NO_COVERED_RETIMING = 7


@dataclass(frozen=True)
class CTBRReferenceConfig:
    max_backup_steps: int = 80
    speed_samples: int = 9
    thrust_reserve_fraction: float = 0.1
    stop_speed: float = 1e-3
    stop_tilt: float = 1e-3
    stop_acceleration: float = 1e-3

    def __post_init__(self):
        if (
            isinstance(self.max_backup_steps, bool)
            or not isinstance(self.max_backup_steps, int)
            or self.max_backup_steps < 2
        ):
            raise ValueError("max_backup_steps must be an integer >= 2")
        if isinstance(self.speed_samples, bool) or not isinstance(self.speed_samples, int) or self.speed_samples < 2:
            raise ValueError("speed_samples must be an integer >= 2")
        if not np.isfinite(self.thrust_reserve_fraction) or not 0 < self.thrust_reserve_fraction < 1:
            raise ValueError("thrust_reserve_fraction must be strictly between zero and one")
        if any(not np.isfinite(x) or x <= 0 for x in (self.stop_speed, self.stop_tilt, self.stop_acceleration)):
            raise ValueError("Stopping tolerances must be finite and positive")


def _rotation(q):
    w, x, y, z = q
    return np.array(
        [
            [w * w + x * x - y * y - z * z, 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), w * w - x * x + y * y - z * z, 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), w * w - x * x - y * y + z * z],
        ]
    )


def _acceleration(state, control, model):
    return _rotation(state[3:7])[:, 2] * (control[0] / model.mass) - np.array([0.0, 0.0, model.gravity])


def _ctbr_step(state, control, model):
    """NumPy parity implementation of CTBRDynamics.step, including raw q semantics."""
    q, omega = state[3:7], control[1:]
    angle2 = np.dot(omega, omega) * (0.5 * model.dt) ** 2
    angle = np.sqrt(max(angle2, 1e-16))
    cosine = 1 - angle2 / 2 + angle2**2 / 24 if angle2 < 1e-8 else np.cos(angle)
    sinc = 1 - angle2 / 6 + angle2**2 / 120 if angle2 < 1e-8 else np.sin(angle) / angle
    increment = omega * (0.5 * model.dt * sinc)
    next_q = np.r_[
        q[0] * cosine - np.dot(q[1:], increment), q[0] * increment + cosine * q[1:] + np.cross(q[1:], increment)
    ]
    next_q /= np.linalg.norm(next_q) + 1e-8
    return np.r_[state[:3] + model.dt * state[7:], next_q, state[7:] + model.dt * _acceleration(state, control, model)]


def _policy_table(model, options):
    rate = min(model.body_rate_limits[:2])
    force_limit = model.specific_thrust_limits[1]
    reserved = model.gravity + (1 - options.thrust_reserve_fraction) * (force_limit - model.gravity)
    angles = np.linspace(0.0, np.arccos(model.gravity / reserved), 129)
    # Approximate policy design only. The complete canonical rollout is checked
    # independently; interpolation in this table never supplies a certificate.
    offsets = np.arange(int(np.ceil(angles[-1] / (rate * model.dt))) + 1) * rate * model.dt
    impulses = model.gravity * model.dt * np.tan(np.maximum(angles[:, None] - offsets, 0.0)).sum(axis=1)
    return rate, angles, impulses


def _brake_control(state, model, table):
    rate, angles, impulses = table
    rotation = _rotation(state[3:7])
    body_z = rotation[:, 2]
    az = np.clip(-state[9] / model.dt, model.acceleration_lower[2], model.acceleration_upper[2])
    thrust = model.mass * (model.gravity + az) / body_z[2] if body_z[2] > 0 else 0.0
    thrust = float(np.clip(thrust, *(model.mass * np.asarray(model.specific_thrust_limits))))
    acceleration = body_z * (thrust / model.mass) - np.array([0.0, 0.0, model.gravity])
    next_horizontal = state[7:9] + model.dt * acceleration[:2]
    speed = np.linalg.norm(next_horizontal)
    theta = np.interp(speed, impulses, angles)
    desired_z = (
        np.r_[-next_horizontal * np.sin(theta) / speed, np.cos(theta)] if speed > 0 else np.array([0.0, 0.0, 1.0])
    )
    target_body = (rotation / np.dot(state[3:7], state[3:7])).T @ desired_z
    axis = np.array([-target_body[1], target_body[0], 0.0])
    length = np.linalg.norm(axis)
    angle = np.arctan2(length, target_body[2])
    if length <= 1e-15:
        axis = np.array([1.0, 0.0, 0.0]) if target_body[2] < 0 else np.zeros(3)
    else:
        axis /= length
    omega = axis * min(angle / model.dt, rate)
    omega = np.clip(omega, -np.asarray(model.body_rate_limits), model.body_rate_limits)
    return np.r_[thrust, omega]


def _inside(polytope, position):
    return bool(np.all(polytope[:, :3] @ position + polytope[:, 3] <= 0.0))


def _horizontal_radius(polytope, position):
    """Largest centered horizontal disc at the actual height; caps do not limit it."""
    norms = np.hypot(polytope[:, 0], polytope[:, 1])
    keep = norms > 0
    slack = -(polytope[:, :3] @ position + polytope[:, 3])
    return float(np.min(slack[keep] / norms[keep])) if keep.any() else np.inf


def _backup(initial, polytopes, model, options, table, *, require_horizontal_reserve=False):
    state = initial.copy()
    states, controls, accelerations = [state.copy()], [], []
    geometric_survivors = np.array([_inside(p, state[:3]) for p in polytopes])
    survivors = geometric_survivors.copy()
    radii = np.array([_horizontal_radius(p, state[:3]) for p in polytopes]) if require_horizontal_reserve else None
    physical_max, horizontal_max = 0.0, 0.0
    reason, stopped = "no_containing_polytope", 0
    for _ in range(options.max_backup_steps if survivors.any() else 0):
        control = _brake_control(state, model, table)
        acceleration = _acceleration(state, control, model)
        following = _ctbr_step(state, control, model)
        controls.append(control)
        accelerations.append(acceleration)
        states.append(following.copy())
        horizontal_max = max(horizontal_max, float(np.linalg.norm(following[:2] - initial[:2])))
        limits = np.r_[model.mass * np.asarray(model.specific_thrust_limits)[:1], -np.asarray(model.body_rate_limits)]
        caps = np.r_[model.mass * np.asarray(model.specific_thrust_limits)[1:], model.body_rate_limits]
        physical_max = max(
            physical_max,
            float(
                np.max(
                    np.r_[
                        limits - control,
                        control - caps,
                        np.asarray(model.acceleration_lower) - acceleration,
                        acceleration - np.asarray(model.acceleration_upper),
                        np.asarray(model.velocity_lower) - following[7:],
                        following[7:] - np.asarray(model.velocity_upper),
                    ]
                )
            ),
        )
        if not np.isfinite(following).all() or not np.isfinite(acceleration).all() or physical_max > 0:
            reason = "physical_bound_or_nonfinite"
            break
        geometric_survivors &= [_inside(p, following[:3]) for p in polytopes]
        if not geometric_survivors.any():
            reason = "left_every_single_polytope"
            break
        survivors = (
            geometric_survivors if not require_horizontal_reserve else geometric_survivors & (horizontal_max <= radii)
        )
        if not survivors.any():
            reason = "horizontal_reserve_not_verified"
            break
        unit_z = _rotation(following[3:7])[:, 2] / np.dot(following[3:7], following[3:7])
        tilt = np.arctan2(np.linalg.norm(unit_z[:2]), unit_z[2])
        terminal = (
            np.linalg.norm(following[7:]) <= options.stop_speed
            and tilt <= options.stop_tilt
            and np.linalg.norm(acceleration) <= options.stop_acceleration
        )
        stopped = stopped + 1 if terminal else 0
        state = following
        if stopped >= 2 and len(controls) >= model.horizon:
            reason = "verified_local_brake"
            break
        reason = "step_limit"
    certified = reason == "verified_local_brake"
    chosen = int(np.flatnonzero(survivors)[0]) if certified else -1
    trajectory = np.asarray(states)
    return {
        "scope": "canonical finite braking witness inside one unchanged current polytope; no Newton or recursive guarantee",
        "certified": certified,
        "reason": reason,
        "corridor": chosen,
        "steps": len(controls),
        "states": trajectory,
        "controls": np.asarray(controls).reshape(-1, 4),
        "accelerations": np.asarray(accelerations).reshape(-1, 3),
        "horizontal_reserve_required": require_horizontal_reserve,
        "horizontal_clearances": radii,
        "horizontal_max_excursion": horizontal_max,
        "max_physical_violation": physical_max,
        "terminal_speed": float(np.linalg.norm(trajectory[-1, 7:])),
        "max_plane_violation": max(
            0.0, float(np.max(trajectory[:, :3] @ polytopes[chosen][:, :3].T + polytopes[chosen][:, 3]))
        )
        if certified
        else None,
    }


def _segment_interval(polytope, a, b):
    lower, upper = 0.0, 1.0
    origin, slope = polytope[:, :3] @ a + polytope[:, 3], polytope[:, :3] @ (b - a)
    for value, direction in zip(origin, slope, strict=True):
        if direction > 0:
            upper = min(upper, -value / direction)
        elif direction < 0:
            lower = max(lower, -value / direction)
        elif value > 0:
            return None
    return (lower, upper) if lower <= upper else None


def _covered(polytopes, a, b):
    intervals = sorted(x for p in polytopes if (x := _segment_interval(p, a, b)) is not None)
    end = 0.0
    for lower, upper in intervals:
        if lower > end:
            return False
        end = max(end, upper)
    return end >= 1.0


def _schedule(polytopes, positions, actual_position):
    if not all(
        _covered(polytopes, a, b) for a, b in zip(np.vstack((actual_position, positions[:-1])), positions, strict=True)
    ):
        return None
    inside = np.array([[_inside(p, x) for p in polytopes] for x in positions])
    if len(polytopes) == 1:
        return np.zeros(len(positions), dtype=np.int64) if inside[:, 0].all() else None
    # Latest valid 0->1 transition; all rows of the chosen original polytope remain.
    for switch in range(len(positions), -1, -1):
        indices = (np.arange(len(positions)) >= switch).astype(np.int64)
        if inside[np.arange(len(positions)), indices].all():
            return indices
    return None


def adapt_ctbr_reference(context, ctbr_state, model, *, path=None, nearest=None, options=None):
    """Return an owned adapted context or the exact original object, plus audit data.

    The maximum is over a fixed, reported finite speed grid. Virtual velocities
    retain measured lateral and vertical components. Only the path-parallel
    horizontal component changes; measured state is never edited.
    """
    options = CTBRReferenceConfig() if options is None else options
    state = np.asarray(ctbr_state, dtype=np.float64)
    reference = np.asarray(context.reference)
    if state.shape != (10,) or not np.isfinite(state).all() or abs(np.linalg.norm(state[3:7]) - 1) > 1e-4:
        raise ValueError("Expected finite near-unit WXYZ CTBR state[10]")
    if reference.shape != (model.horizon, 9) or not np.isfinite(reference).all():
        raise ValueError("Expected finite reference[horizon,9]")
    measured = np.asarray(context.state)
    if measured.shape != (9,) or not np.issubdtype(measured.dtype, np.floating) or not np.isfinite(measured).all():
        raise ValueError("Expected finite floating planning state[9]")
    # IPC subtracts environment origins in its observation dtype; the CTBR
    # adapter subtracts after promoting to float64. Compare at the observation
    # precision while keeping the full measured CTBR state for every rollout.
    if not np.array_equal(measured[:3], state[:3].astype(measured.dtype)) or not np.array_equal(
        measured[3:6], state[7:].astype(measured.dtype)
    ):
        raise ValueError("Planning and CTBR contexts must share the measured position and velocity")
    diagnostics = dict(
        zip(
            REFERENCE_DIAGNOSTIC_FIELDS,
            (
                REFERENCE_GOVERNOR_VERSION,
                False,
                int(ReferenceReason.NO_FORWARD_PATH),
                False,
                False,
                False,
                1.0,
                0.0,
                0.0,
                0,
                0,
                -1,
            ),
            strict=True,
        )
    )
    diagnostics.update(
        scope="finite local braking and virtual horizontal-reserve policy; no returned-NLP, Newton, or future-obstacle guarantee",
        original_reference=reference.copy(),
        adapted_reference=reference.copy(),
        evaluated_speeds=[],
        actual_backup_matches_returned_schedule=False,
    )
    if not context.valid:
        diagnostics["reason_code"] = int(ReferenceReason.INVALID_CONTEXT)
        return context, diagnostics
    polytopes, original_indices = [], []
    if len(context.corridors) != model.horizon:
        raise ValueError("One original corridor slot is required per reference")
    for faces in context.corridors:
        faces = np.asarray(faces)
        if faces.ndim != 2 or faces.shape[1] != 4 or not np.isfinite(faces).all():
            raise ValueError("Corridor rows must be finite [N,4]")
        if not len(faces):
            diagnostics["reason_code"] = int(ReferenceReason.NO_CONTAINING_CORRIDOR)
            return context, diagnostics
        index = next((i for i, p in enumerate(polytopes) if np.array_equal(p, faces)), len(polytopes))
        if index == len(polytopes):
            polytopes.append(faces)
        original_indices.append(index)
    if len(polytopes) > 2:
        raise ValueError("Expected at most two unchanged original polytopes")
    diagnostics.update(
        original_corridor_indices=np.asarray(original_indices), adapted_corridor_indices=np.asarray(original_indices)
    )
    table = _policy_table(model, options)
    actual = _backup(state, polytopes, model, options, table)
    actual["state_kind"] = "measured"
    diagnostics.update(
        actual_backup=actual, actual_backup_certified=actual["certified"], actual_backup_steps=actual["steps"]
    )
    if not actual["certified"]:
        diagnostics["reason_code"] = int(ReferenceReason.ACTUAL_BACKUP_NOT_CERTIFIED)
        return context, diagnostics
    if path is None or nearest is None:
        return context, diagnostics
    path = np.asarray(path, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 3 or not np.isfinite(path).all() or not 0 <= nearest < len(path):
        raise ValueError("Expected finite original path[N,3] and valid nearest index")
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    sample_indices, cursor = [], int(nearest)
    for point in reference[:, :3]:
        matches = np.flatnonzero((path[cursor:] == point).all(axis=1))
        if not len(matches):
            return context, diagnostics
        cursor += int(matches[0])
        sample_indices.append(cursor)
    nominal_arc = arc[sample_indices]
    forward = np.diff(path[int(nearest) :], axis=0)[:, :2]
    nonzero = np.flatnonzero(np.linalg.norm(forward, axis=1) > 0)
    if not len(nonzero):
        return context, diagnostics
    tangent = forward[nonzero[0]] / np.linalg.norm(forward[nonzero[0]])
    nominal_speed = float(np.max(np.diff(nominal_arc), initial=0.0) / model.dt)
    diagnostics.update(nominal_speed=nominal_speed, selected_speed=nominal_speed, forward_tangent_xy=tangent.copy())
    lateral = state[7:9] - np.dot(state[7:9], tangent) * tangent
    for speed in np.linspace(nominal_speed, 0.0, options.speed_samples):
        virtual = state.copy()
        virtual[7:9] = lateral + speed * tangent
        backup = _backup(virtual, polytopes, model, options, table, require_horizontal_reserve=True)
        backup["state_kind"] = "virtual_path_speed_at_measured_pose"
        diagnostics["evaluated_speeds"].append(
            {
                "speed": float(speed),
                "certified": backup["certified"],
                "reason": backup["reason"],
                "steps": backup["steps"],
                "horizontal_clearances": backup["horizontal_clearances"].copy(),
                "horizontal_max_excursion": backup["horizontal_max_excursion"],
                "corridor": backup["corridor"],
            }
        )
        if speed == nominal_speed:
            diagnostics["nominal_backup_certified"] = backup["certified"]
        if not backup["certified"]:
            continue
        new_arc = nominal_arc.copy()
        if speed < nominal_speed:
            for i in range(1, len(new_arc)):
                new_arc[i] = min(nominal_arc[i], new_arc[i - 1] + speed * model.dt)
        positions = (
            reference[:, :3].copy()
            if speed == nominal_speed
            else np.stack([np.interp(new_arc, arc, path[:, j]) for j in range(3)], axis=1)
        )
        schedule = _schedule(polytopes, positions, state[:3])
        # Check intermediate original path vertices too: no shortcut around a bend.
        vertices = np.vstack((positions[0], path[(arc > new_arc[0]) & (arc < new_arc[-1])], positions[-1]))
        if schedule is None or not all(
            _covered(polytopes, a, b) for a, b in zip(vertices[:-1], vertices[1:], strict=True)
        ):
            continue
        original_covered = all(_inside(polytopes[i], p) for i, p in zip(original_indices, positions, strict=True))
        identity = speed == nominal_speed and original_covered
        if identity:
            schedule = np.asarray(original_indices, dtype=np.int64)
        # Interpolated path points were checked in float64. Preserve those
        # values rather than rounding them into a possibly float32 source array.
        adapted = reference.copy() if identity else np.asarray(reference, dtype=np.float64).copy()
        adapted[:, :3] = positions
        if not identity:
            adapted[:, 3:6] = np.diff(np.vstack((state[:3], positions)), axis=0) / model.dt
        corridors = [polytopes[i] for i in schedule]
        matches = len(actual["states"]) > model.horizon and all(
            _inside(faces, position)
            for faces, position in zip(corridors, actual["states"][1 : model.horizon + 1, :3], strict=True)
        )
        diagnostics.update(
            applied=not identity,
            reason_code=int(ReferenceReason.IDENTITY_CERTIFIED if identity else ReferenceReason.ADAPTED_LOCAL_BACKUP),
            selected_backup_certified=True,
            selected_speed=float(speed),
            reference_scale=float(speed / nominal_speed) if nominal_speed else 1.0,
            selected_backup_steps=backup["steps"],
            containing_corridor=backup["corridor"],
            selected_backup=backup,
            adapted_reference=adapted.copy(),
            adapted_corridor_indices=schedule.copy(),
            actual_backup_matches_returned_schedule=bool(matches),
        )
        if identity:
            return context, diagnostics
        owned_polytopes = [p.copy() for p in polytopes]
        return replace(
            context,
            state=measured.copy(),
            reference=adapted,
            corridors=[owned_polytopes[i] for i in schedule],
        ), diagnostics
    diagnostics["reason_code"] = int(
        ReferenceReason.NO_VIRTUAL_BACKUP
        if not any(x["certified"] for x in diagnostics["evaluated_speeds"])
        else ReferenceReason.NO_COVERED_RETIMING
    )
    return context, diagnostics
