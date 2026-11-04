"""Numerical deployment contracts; live MARSIM runs remain the acceptance test."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from isaaclab_ipc.corridor_storage import expand_corridors
from isaaclab_ipc.frontend import PlanningContext
from isaaclab_ipc.ros_policy import observation


def test_ros_features_equal_the_actual_training_expression():
    """Evaluate the source training expression without loading a simulator app."""
    import isaaclab_ipc

    physical = torch.arange(18, dtype=torch.float32).reshape(1, 18) / 7
    goal = torch.tensor([[1.1, -2.3, 4.7]], dtype=torch.float32)
    reference = torch.arange(135, dtype=torch.float64).reshape(1, 15, 9) / 13
    face = np.array([[1., 2., 0., -3.], [-1., 0., 0., -1.]])
    corridors = [face if i % 3 else np.empty((0, 4)) for i in range(15)]
    margins = [[min(-(f[:3] @ reference[0, i, :3].numpy() + f[3]) / np.linalg.norm(f[:3])
                    for f in faces) if len(faces) else 10. for i, faces in enumerate(corridors)]]
    braking = [[True]]

    def tensor(value, dtype=torch.float64):
        return torch.as_tensor(np.asarray(value), dtype=dtype)

    source = Path(isaaclab_ipc.__file__).parent / "tasks/mpc_env.py"
    tree = ast.parse(source.read_text())
    expression = next(n.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == "features" for target in n.targets))
    expected = eval(compile(ast.Expression(expression), str(source), "eval"), dict(
        torch=torch, physical=physical, reference=reference, self=SimpleNamespace(goal=goal),
        tensor=tensor, margins=margins, braking=braking,
    ))
    context = PlanningContext(physical[0, :9].double().numpy(), reference[0].numpy(), corridors, True)
    actual = observation(physical.numpy(), goal[0].numpy(), context, braking=True)
    assert actual["policy"].shape == (1, 124)
    torch.testing.assert_close(actual["policy"], expected, rtol=0, atol=0)
    torch.testing.assert_close(actual["mpc_state"], physical[:, :9].double(), rtol=0, atol=0)
    torch.testing.assert_close(actual["reference"], reference, rtol=0, atol=0)
    assert bool(actual["braking"][0, 0])
    assert bool(actual["sfc_valid"][0, 0])
    dense, mask = expand_corridors(actual)
    for i, faces in enumerate(corridors):
        np.testing.assert_array_equal(dense[0, i, mask[0, i]].numpy(), faces[::-1])


def test_empty_frontend_context_preserves_recovery_inputs():
    physical = np.r_[[1., 2., 3., .4, .5, .6, .7, .8, .9], np.eye(3).ravel()]
    context = PlanningContext(physical[:9], np.zeros((15, 9)), [], False)
    target = np.array([1., 2., 4.])
    obs = observation(physical, [5., 6., 7.], context, yaw_target=.3, climb_target=target)
    assert not bool(obs["sfc_valid"][0, 0])
    assert bool(obs["climb_active"][0, 0])
    np.testing.assert_array_equal(obs["climb_target"][0].numpy(), target)
    assert float(obs["yaw_target"][0, 0]) == .3
    assert obs["plane_mask"].sum() == 0
    assert obs["corridor_index"].shape == (1, 15)
    assert torch.all(obs["corridor_index"] == -1)
