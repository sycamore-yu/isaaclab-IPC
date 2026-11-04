"""Execute an IPC jerk sequence between asynchronous ROS planning updates."""

from collections import deque

import numpy as np
import torch

from .execution import brake_jerk, safe_climb_jerk
from .mpc import dynamics
from .rl.actor import flatness_torch


def check_plan_constraints(plan, state, mpc):
    """Recheck the solved jerk against original constraints at the execution PVA.

    MPC.problem clips its input state. Build its constraints at zero and add the
    actual state's affine offset instead, so out-of-bound feedback is not hidden.
    This checks feasibility only; the learned costs and QP solution stay intact.
    """
    empty = np.empty((0, 4))
    corridors = [plan["corridors"][i] if i >= 0 else empty for i in plan["corridor_index"]]
    _, _, matrix, lower, upper, _ = mpc.problem(np.zeros(9), np.zeros((mpc.n, 9)), corridors)
    offset = mpc.M @ state
    shift = [np.zeros(3 * mpc.n), offset[mpc.ai], offset[mpc.vi]]
    for i, faces in enumerate(corridors):
        if len(faces):
            shift.append(faces[::-1, :3] @ offset[9 * i : 9 * i + 3])
    shift = np.concatenate(shift)
    lower, upper = lower - shift, upper - shift
    value = matrix @ plan["sequence"].ravel()
    violations = np.maximum(np.maximum(lower - value, value - upper), 0)
    residual = float(violations.max())
    # OSQP's primal tolerance uses max(||A u||inf, ||z||inf), with z in [l,u].
    # The actor uses only eps_abs for primal feasibility (eps_rel is for dual).
    scale = max(np.abs(value).max(), np.abs(np.clip(value, lower, upper)).max())
    tolerance = float(plan["constraint_eps_abs"] + plan["constraint_eps_rel"] * scale)
    row = int(violations.argmax())
    group = min(row // (3 * mpc.n), 3)
    feasible = residual <= tolerance
    return dict(feasible=feasible, violation=residual, tolerance=tolerance,
                constraint=None if feasible else ("jerk", "acceleration", "velocity", "sfc")[group])


def accept_plan(plan, state, stamp_ns, mpc):
    """Anchor execution in the latest PVA and retain the original MPC knots."""
    accepted = dict(plan, accepted_stamp_ns=stamp_ns)
    states = np.empty((len(plan["sequence"]) + 1, 9), dtype=np.float64)
    states[0] = state
    for i, jerk in enumerate(plan["sequence"]):
        states[i + 1] = mpc.a @ states[i] + mpc.b @ jerk
    accepted["states"] = states
    return accepted


def sequence_reference(plan, stamp_ns, mpc):
    """Sample inside a 0.1 s model interval without advancing measured a again.

    Each knot is the original discrete prediction. Between knots, evaluate the
    same constant-jerk dynamics for the remaining fraction of that interval.
    """
    delay_ns = round(float(mpc.cfg.params["ctrl_delay"]) * 1e9)
    elapsed_ns = max(0, stamp_ns - plan["accepted_stamp_ns"] + delay_ns)
    index, fraction_ns = divmod(elapsed_ns, round(mpc.dt * 1e9))
    if fraction_ns == 0:
        return plan["states"][index].copy()
    drag = [mpc.cfg.params["mpc"]["D_" + axis] for axis in "xyz"]
    a, b = dynamics(fraction_ns * 1e-9, drag)
    return a @ plan["states"][index] + b @ plan["sequence"][index]


def select_jerk(plan, state, stamp_ns, mpc):
    """Execute a new sequence from acceptance, preserving source-state validity.

    Planning latency is not execution of the new sequence. Its first jerk is
    therefore selected when accepted, just as in Planner.consume_state. The
    source timestamp still limits validity to the original prediction horizon.
    """
    state = np.asarray(state, dtype=np.float64).copy()
    state[3:] = np.clip(state[3:], mpc.lower, mpc.upper)
    index = None
    expired = False
    if plan is not None:
        delay_ns = round(float(mpc.cfg.params["ctrl_delay"]) * 1e9)
        elapsed_ns = max(0, stamp_ns - plan["accepted_stamp_ns"] + delay_ns)
        index = elapsed_ns // round(mpc.dt * 1e9)
        horizon_ns = len(plan["sequence"]) * round(mpc.dt * 1e9)
        expired = stamp_ns - plan["source_stamp_ns"] >= horizon_ns or index >= len(plan["sequence"])
    braking = plan is None or expired or plan["braking"] or not plan["sfc_valid"]
    if braking:
        target = None if plan is None else plan["climb_target"]
        if target is not None:
            jerk = safe_climb_jerk(state, target[2], mpc.lower, mpc.upper, mpc.jlow, mpc.jhigh, mpc.dt)
        else:
            jerk = brake_jerk(state, mpc.lower, mpc.upper, mpc.jlow, mpc.jhigh, mpc.dt)
    else:
        jerk = plan["sequence"][index].copy()
    return jerk, index, bool(braking), bool(expired)


def ctbr_from_jerk(state, rotation, jerk, yaw, mpc, thrust_limits, rate_limit):
    """Original one-step feedback for recovery, whose jerk is recomputed each tick."""
    state = np.asarray(state, dtype=np.float64).copy()
    state[3:] = np.clip(state[3:], mpc.lower, mpc.upper)
    predicted = mpc.a @ state + mpc.b @ jerk
    return ctbr_from_reference(predicted[6:], rotation, jerk, yaw, thrust_limits, rate_limit)


def ctbr_from_reference(acceleration, rotation, jerk, yaw, thrust_limits, rate_limit):
    """Use the original flatness map with reference a/j and current attitude."""
    with torch.inference_mode():
        return flatness_torch(
            torch.as_tensor(acceleration[None], dtype=torch.float64), torch.as_tensor(jerk[None], dtype=torch.float64),
            torch.as_tensor(rotation[None], dtype=torch.float64), torch.as_tensor([yaw], dtype=torch.float64),
            thrust_limits, rate_limit,
        )[0].numpy()


class IdealCommand:
    """Advance the last direct PVA command over the actual publication interval.

    PerfectDrone holds the most recent PositionCommand. Its odometry can be
    slower than command publication, so restarting integration from every old
    odometry sample would lose commanded time and distance.
    """

    def __init__(self, drag, history_size=256):
        self.drag = drag
        self.state = None
        self.stamp_ns = None
        self.jerk = np.zeros(3)
        self.pending = deque(maxlen=history_size)

    def observed_acceleration(self, position_velocity):
        """Pair PerfectDrone's copied P/V with the same executed command's a.

        Odometry may arrive after newer commands were published. PerfectDrone
        copies all PositionCommand fields, so exact P/V identifies that command.
        Keep the matched command for repeated odometry and discard older ones.
        """
        for newer, state in enumerate(reversed(self.pending)):
            if np.array_equal(state[:6], position_velocity):
                for _ in range(len(self.pending) - newer - 1):
                    self.pending.popleft()
                return state[6:].copy()
        raise ValueError("Ideal odometry P/V does not match a retained PositionCommand")

    def advance(self, measured_state, jerk, stamp_ns):
        if self.state is None:
            self.state = np.asarray(measured_state, dtype=np.float64).copy()
        else:
            dt = (stamp_ns - self.stamp_ns) * 1e-9
            a, b = dynamics(dt, self.drag)
            self.state = a @ self.state + b @ self.jerk
        self.jerk = np.asarray(jerk).copy()
        self.stamp_ns = stamp_ns
        self.pending.append(self.state.copy())
        return self.state.copy()
