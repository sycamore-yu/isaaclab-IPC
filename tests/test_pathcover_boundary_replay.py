"""Frozen reference problems keep their original initial-boundary semantics."""

import importlib
from pathlib import Path

import numpy as np
import pytest

from isaaclab_ipc.pathcover_native import PathCoverConfig


@pytest.fixture
def replay_module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("replay_pathcover_gcopter")


def test_collective_replay_serializes_physical_bounds_and_measured_projection(replay_module):
    config = PathCoverConfig(thrust_min=0.0, nominal_thrust_min=2.1, thrust_max=22.0)
    initial = np.column_stack(([1.0, 2.0, 3.0], [0.2, -0.1, 0.4], [2.0, -1.0, 0.3]))
    problem = {"initial_pva": initial, "terminal_pva": np.zeros((3, 3)), "corridors": ()}
    fixed = np.fromstring(replay_module.serialized_problem(problem, config), sep=" ")
    axis = np.array([0.0, 0.6, 0.8])
    problem["initial_boundary"] = {"mode": "ctbr_collective", "thrust_axis": axis.tolist()}
    collective = np.fromstring(replay_module.serialized_problem(problem, config), sep=" ")
    np.testing.assert_array_equal(collective[:-7], fixed)
    expected_guess = config.mass * float((initial[:, 2] + [0.0, 0.0, config.gravity]) @ axis)
    np.testing.assert_array_equal(collective[-7:], [1, *axis, 0.0, 22.0, expected_guess])
    np.testing.assert_array_equal(initial[:, 2], [2.0, -1.0, 0.3])


def test_historical_replay_does_not_infer_collective_mode_from_measured_state(replay_module):
    config = PathCoverConfig(thrust_min=0.0, nominal_thrust_min=2.1, thrust_max=22.0)
    problem = {"initial_pva": np.zeros((3, 3)), "terminal_pva": np.zeros((3, 3)), "corridors": ()}
    original = replay_module.serialized_problem(problem, config)
    problem["initial_boundary"] = {"mode": "fixed_pva", "thrust_axis": None}
    assert replay_module.serialized_problem(problem, config) == original


def test_official_replay_rejects_collective_contract_before_running_solver(replay_module, monkeypatch):
    def unexpected_run(*args, **kwargs):
        pytest.fail("A different fixed-PVA problem must not run as an official comparison")

    monkeypatch.setattr(replay_module.subprocess, "run", unexpected_run)
    with pytest.raises(ValueError, match="no CTBR collective boundary"):
        replay_module.replay(Path("unused"), "", {"initial_boundary": {"mode": "ctbr_collective"}},
                             PathCoverConfig(), False)


@pytest.mark.parametrize("collective", [False, True])
def test_continuous_position_cost_has_explicit_replay_tail(replay_module, collective):
    from dataclasses import replace

    config = PathCoverConfig(thrust_min=0.0, nominal_thrust_min=2.1, thrust_max=22.0)
    problem = {"initial_pva": np.zeros((3, 3)), "terminal_pva": np.zeros((3, 3)), "corridors": ()}
    if collective:
        problem["initial_boundary"] = {"mode": "ctbr_collective", "thrust_axis": [0.0, 0.0, 1.0]}
    historical = np.fromstring(replay_module.serialized_problem(problem, config), sep=" ")
    selected = replace(config, position_penalty_aggregation="continuous_max")
    current = np.fromstring(replay_module.serialized_problem(problem, selected), sep=" ")
    tail = [1] if collective else [0, 1]
    np.testing.assert_array_equal(current[:-len(tail)], historical)
    np.testing.assert_array_equal(current[-len(tail):], tail)
    assert config.position_penalty_aggregation == "integral"


def test_historical_position_cost_does_not_follow_ctbr_factory_default(replay_module):
    recorded = PathCoverConfig(position_penalty_aggregation="continuous_max").native_options()
    assert replay_module.config_from_snapshot(recorded).position_penalty_aggregation == "continuous_max"
    del recorded["position_penalty_aggregation"]
    assert replay_module.config_from_snapshot(recorded).position_penalty_aggregation == "integral"
    assert "position_penalty_aggregation" not in recorded


@pytest.mark.parametrize("mode", ["piece_max", "", None])
def test_unknown_position_cost_is_not_silently_replaced(mode):
    with pytest.raises(ValueError, match="position_penalty_aggregation"):
        PathCoverConfig(position_penalty_aggregation=mode)


def test_official_replay_rejects_continuous_cost_before_running_solver(replay_module, monkeypatch):
    def unexpected_run(*args, **kwargs):
        pytest.fail("A different integral-cost problem must not run as an official comparison")

    monkeypatch.setattr(replay_module.subprocess, "run", unexpected_run)
    config = PathCoverConfig(position_penalty_aggregation="continuous_max")
    with pytest.raises(ValueError, match="no continuous_max position cost"):
        replay_module.replay(Path("unused"), "", {}, config, False)


@pytest.mark.parametrize("entry", ["whole_chain", "gcopter"])
def test_other_official_oracles_reject_unavailable_cost(replay_module, monkeypatch, entry):
    from types import SimpleNamespace

    def unexpected_call(*args, **kwargs):
        pytest.fail("The unsupported cost must be detected before building or running the oracle")

    monkeypatch.setattr(replay_module.subprocess, "check_output", unexpected_call)
    config = PathCoverConfig(position_penalty_aggregation="continuous_max")
    with pytest.raises(ValueError, match="no continuous_max position cost"):
        if entry == "whole_chain":
            module = importlib.import_module("compare_pathcover_upstream")
            module.serialized_input(np.empty((0, 3)), np.zeros((3, 3)), np.zeros(3), config)
        else:
            module = importlib.import_module("validate_pathcover_native")
            module.official_gcopter(Path("unused"), SimpleNamespace(metrics={}), None, config)
