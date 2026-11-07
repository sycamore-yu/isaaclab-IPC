"""Trace snapshots retain the actual frontend boundary without replaying it."""

import ast
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


@pytest.mark.parametrize("fails", [False, True])
def test_trace_freezes_new_initial_boundary_without_changing_plan_call(tmp_path, fails):
    # Compile the unchanged nested trace wrapper without starting an Isaac app.
    source = Path(__file__).resolve().parents[1] / "scripts/trace_ctbr_newton.py"
    main = next(node for node in ast.parse(source.read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name == "main")
    wrapper = next(node for node in main.body if isinstance(node, ast.FunctionDef) and node.name == "plan")
    factory = ast.parse("def factory():\n    plan_count = 0\n    return plan\n")
    factory.body[0].body.insert(1, wrapper)

    @dataclass
    class State:
        position: np.ndarray
        velocity: np.ndarray
        acceleration: np.ndarray
        yaw: float

    @dataclass
    class Config:
        mass: float = .75

    state = State(np.array([1., 2., 3.]), np.array([.1, .2, .3]), np.array([.4, .5, .6]), .37)
    axis = np.array([0., .6, .8])
    kwargs = dict(initial_motion_seed=np.arange(9.).reshape(3, 3),
                  initial_motion_tip_segment=np.arange(6.).reshape(2, 3), initial_thrust_axis=axis)
    boundary = dict(mode="ctbr_collective", measured_acceleration=state.acceleration.tolist(),
                    reference_acceleration=[1., 2., 3.], thrust_axis=axis.tolist(), force_n=4.)
    result = SimpleNamespace(metrics={"initial_boundary": boundary})
    calls = []

    def original_plan(frontend, points, received_state, goal, **received):
        calls.append(received)
        assert received_state is state
        assert received_state.yaw == .37
        for name, value in kwargs.items():
            assert received[name] is value
        if fails:
            raise RuntimeError("known diagnostic failure")
        return result

    frontend = SimpleNamespace(config=Config(), failed_problem=lambda: {"initial_boundary": boundary})
    namespace = dict(np=np, torch=torch, asdict=asdict, deepcopy=deepcopy,
                     args=SimpleNamespace(output=tmp_path), reset_generations={}, original_plan=original_plan)
    exec(compile(ast.fix_missing_locations(factory), str(source), "exec"), namespace)
    traced = namespace["factory"]()
    if fails:
        with pytest.raises(RuntimeError, match="known diagnostic failure"):
            traced(frontend, np.ones((1, 3)), state, np.array([8., 0., 1.]), **kwargs)
        stored = torch.load(tmp_path / "failed-frontend-call.pt", weights_only=False)
        assert stored["gcopter_problem"]["initial_boundary"] == boundary
    else:
        assert traced(frontend, np.ones((1, 3)), state, np.array([8., 0., 1.]), **kwargs) is result
        stored = torch.load(tmp_path / "frontend-inputs/00001.pt", weights_only=False)
        first = torch.load(tmp_path / "first-frontend-input.pt", weights_only=False)
        assert stored["initial_boundary"] == first["initial_boundary"] == boundary
    assert len(calls) == 1
    assert stored["current_state"]["yaw"] == state.yaw
    np.testing.assert_array_equal(stored["current_state"]["acceleration"], state.acceleration)
    for name, value in kwargs.items():
        np.testing.assert_array_equal(stored[name], value)
