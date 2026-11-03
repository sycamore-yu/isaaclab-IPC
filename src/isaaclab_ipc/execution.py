"""The two supported execution contracts and their fixed recovery policies."""

from enum import StrEnum

import numpy as np


def next_braking(braking, solved, speed, acceleration):
    """Shared scalar/array transition for real execution and its differentiable actor."""
    return (solved ^ True) | (braking & ((speed >= 0.2) | (acceleration >= 0.5)))


def brake_jerk(state, lower, upper, jlow, jhigh, dt):
    desired = (-2.0 * state[..., 3:6]).clip(lower[3:], upper[3:])
    return ((desired - state[..., 6:]) / dt).clip(jlow[:3], jhigh[:3])


def safe_climb_jerk(state, target_z, lower, upper, jlow, jhigh, dt):
    """Retain horizontal braking while controlling altitude toward a checked target."""
    desired = (-2.0 * state[..., 3:6]).clip(lower[3:], upper[3:])
    desired = desired.clone() if hasattr(desired, "clone") else desired.copy()
    desired[..., 2] = (5.0 * (target_z - state[..., 2]) - 3.0 * state[..., 5]).clip(-2.0, 3.0)
    return ((desired - state[..., 6:]) / dt).clip(jlow[:3], jhigh[:3])


class RecoveryPolicy(StrEnum):
    BRAKE_REPLAN = "brake_replan"
    PREVIOUS_SEQUENCE = "previous_sequence"


class ExecutionProfile(StrEnum):
    REAL = "real"
    IDEAL = "ideal"

    @classmethod
    def _missing_(cls, value):
        raise ValueError("Unknown execution profile: " + str(value))

    @property
    def uses_newton(self):
        return self is ExecutionProfile.REAL

    @property
    def recovery(self):
        return RecoveryPolicy.BRAKE_REPLAN if self.uses_newton else RecoveryPolicy.PREVIOUS_SEQUENCE

    def planner_state(self, state):
        """Keep real velocity, or apply upstream perfect_simu without mutating the observation."""
        state = np.asarray(state).copy()
        if not self.uses_newton:
            state[3:6] = 0
        return state

    def validate_evaluation(self, *, forest, goal_x, num_envs):
        if not self.uses_newton and (forest or goal_x is not None or num_envs != 1):
            raise ValueError("Ideal uses one avoid episode with goal (10,0,1); use run_ideal_campaign.py for repeats")
