"""Action conversion keeps its gradient and detached physical diagnostics."""

import pytest
import torch
from tensordict import TensorDict

from isaaclab_ipc.rl.ctbr_actor import CTBRMPCActor
from isaaclab_ipc.rl.ctbr_observation import ctbr_feature_size


def _actor(batch, *, compile_kernels):
    obs = TensorDict({"policy": torch.zeros(batch, ctbr_feature_size(5))}, batch_size=[batch])
    return CTBRMPCActor(obs, {"actor": ["policy"]}, "actor", 4, compile_kernels=compile_kernels)


def _inputs(batch, device="cpu"):
    # Include both saturated controls and force that rounds during float32 conversion.
    rows = torch.tensor(
        [[-0.75, -4.0, 0.4, -2.0], [7.5000005, 0.1, -0.2, 0.3], [30.0, 4.0, -4.0, 2.0]],
        dtype=torch.float64, device=device,
    )
    controls = rows[torch.arange(batch, device=device) % 3, None].repeat(1, 5, 1)
    state = torch.zeros(batch, 10, dtype=torch.float64, device=device)
    state[:, :3] = state.new_tensor([0.2, -0.3, 1.4])
    state[:, 3:7] = state.new_tensor([0.999, 0.01, -0.02, 0.03])
    state[:, 7:] = state.new_tensor([0.7, -0.4, 0.2])
    return controls.requires_grad_(), state.requires_grad_()


def test_eager_action_rounding_clamp_gradient_and_detached_diagnostics():
    actor = _actor(3, compile_kernels=False)
    controls, state = _inputs(3)
    original_controls, original_state = controls.detach().clone(), state.detach().clone()
    mean, applied, predicted, acceleration = actor._action_postprocess(controls, state)
    expected = torch.tensor(
        [[0.0, -3.0, 0.4, -1.5], [10.000001, 0.1, -0.2, 0.3], [29.43, 3.0, -3.0, 1.5]],
        dtype=torch.float32,
    )
    torch.testing.assert_close(mean, expected, atol=0, rtol=0)
    torch.testing.assert_close(applied[:, 0], expected[:, 0].double() * 0.75, atol=0, rtol=0)
    torch.testing.assert_close(applied[:, 1:], expected[:, 1:].double(), atol=0, rtol=0)
    assert mean.requires_grad
    assert all(not value.requires_grad for value in (applied, predicted, acceleration))
    control_grad, state_grad = torch.autograd.grad(mean.sum(), (controls, state), allow_unused=True)
    expected_grad = torch.zeros_like(controls)
    expected_grad[0, 0, 2] = 1.0
    expected_grad[1, 0] = expected_grad.new_tensor([1 / 0.75, 1.0, 1.0, 1.0])
    torch.testing.assert_close(control_grad, expected_grad, atol=0, rtol=0)
    assert state_grad is None
    torch.testing.assert_close(predicted[:, :3], state[:, :3] + 0.1 * state[:, 7:])
    torch.testing.assert_close(predicted[:, 7:], state[:, 7:] + 0.1 * acceleration)
    torch.testing.assert_close(controls, original_controls, atol=0, rtol=0)
    torch.testing.assert_close(state, original_state, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="compiled action helper requires CUDA")
@pytest.mark.parametrize("batch", [1, 16, 64, 256])
def test_compiled_action_helper_migration_checkpoint_and_output_lifetime(batch, tmp_path):
    eager = _actor(batch, compile_kernels=False)
    checkpoint = tmp_path / "actor-state.pt"
    torch.save(eager.state_dict(), checkpoint)
    compiled = _actor(batch, compile_kernels=True)
    compiled.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False), strict=True)
    assert compiled.physical_action_lower.device.type == "cpu"
    compiled.to("cuda")
    assert compiled.physical_action_lower.device.type == "cuda"
    assert compiled.physical_action_upper.device.type == "cuda"

    cpu_controls, cpu_state = _inputs(batch)
    expected = eager._action_postprocess(cpu_controls, cpu_state)
    controls, state = _inputs(batch, "cuda")
    original_controls, original_state = controls.detach().clone(), state.detach().clone()
    actual = compiled._action_postprocess(controls, state)
    assert actual[0].requires_grad
    assert all(not value.requires_grad for value in actual[1:])
    for result, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(result.cpu(), reference, atol=1e-12, rtol=1e-12)
    expected_grad = torch.autograd.grad(expected[0].sum(), cpu_controls)[0]
    actual_grad, state_grad = torch.autograd.grad(actual[0].sum(), (controls, state), allow_unused=True)
    torch.testing.assert_close(actual_grad.cpu(), expected_grad, atol=1e-12, rtol=1e-12)
    # AOTAutograd may materialize a zero for an unused differentiable input.
    assert state_grad is None or torch.count_nonzero(state_grad) == 0

    retained = [value.detach().clone() for value in actual]
    with torch.no_grad():
        for shift in (0.1, -0.2, 0.3):
            compiled._action_postprocess(controls + shift, state + shift)
    for value, saved in zip(actual, retained, strict=True):
        torch.testing.assert_close(value, saved, atol=0, rtol=0)
    torch.testing.assert_close(controls, original_controls, atol=0, rtol=0)
    torch.testing.assert_close(state, original_state, atol=0, rtol=0)
    assert compiled._native_solver is None
