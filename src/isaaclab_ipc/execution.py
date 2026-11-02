"""The two supported execution contracts and their fixed recovery policies."""

from enum import StrEnum

import numpy as np


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
