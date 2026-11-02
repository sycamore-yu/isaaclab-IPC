import json
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from isaaclab_ipc.execution import ExecutionProfile, RecoveryPolicy
from isaaclab_ipc.mpc import Config
from isaaclab_ipc.planner import Planner


@pytest.mark.parametrize(
    "name,newton,recovery",
    [
        ("real", True, RecoveryPolicy.BRAKE_REPLAN),
        ("ideal", False, RecoveryPolicy.PREVIOUS_SEQUENCE),
    ],
)
def test_profile_is_the_single_execution_contract(name, newton, recovery):
    cfg = Config.load("avoid", profile=name)
    assert cfg.profile.uses_newton is newton
    assert cfg.profile.recovery is recovery
    assert json.loads(json.dumps({"profile": cfg.profile})) == {"profile": name}
    assert not {"execution_profile", "perfect_simu", "recovery"} & cfg.params.keys()
    with pytest.raises(FrozenInstanceError):
        cfg.profile = "ideal" if newton else "real"
    with pytest.raises(AttributeError):
        cfg.profile.recovery = "previous_sequence"
    # Numerical tuning remains available; it cannot change the selected recovery policy.
    cfg.params["sfc_dis"] = 0.4
    assert cfg.profile.recovery is recovery


@pytest.mark.parametrize("field", ["execution_profile", "perfect_simu", "recovery"])
def test_reject_execution_metadata_in_numerical_config(field):
    params = dict(Config.load().params)
    params[field] = "conflicting"
    with pytest.raises(ValueError, match="Config.profile"):
        Config(params)


def test_profile_validation_in_load_and_direct_construction():
    assert Config.load().profile is ExecutionProfile.REAL
    with pytest.raises(ValueError, match="Unknown execution profile"):
        Config.load(profile="bad")
    with pytest.raises(ValueError, match="Unknown execution profile"):
        Config({}, profile="bad")


@pytest.mark.parametrize("profile", list(ExecutionProfile))
def test_planner_applies_profile_velocity_without_changing_observation(profile):
    planner = Planner(Config.load("avoid", profile=profile), [8, 0, 1])
    state = np.array([0, 0, 1, 2.0, 0.5, 0, 0, 0, 0])
    original = state.copy()
    problem, _ = planner.prepare(state)
    np.testing.assert_array_equal(state, original)
    np.testing.assert_array_equal(problem[-1][3:6], state[3:6] if profile.uses_newton else np.zeros(3))


@pytest.mark.parametrize("override", [{"forest": True}, {"num_envs": 2}, {"goal_x": 32}])
def test_ideal_evaluation_restrictions_are_central(override):
    args = {"forest": False, "goal_x": None, "num_envs": 1}
    args.update(override)
    with pytest.raises(ValueError, match="Ideal uses one avoid episode"):
        ExecutionProfile.IDEAL.validate_evaluation(**args)
    ExecutionProfile.REAL.validate_evaluation(**args)
