"""Exercise the native actor through saved IPC context and recovery branches."""

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from isaaclab_ipc.corridor_storage import pack_corridors
from isaaclab_ipc.mpc import Config
from isaaclab_ipc.planner import Planner, flatness
from isaaclab_ipc.rl.turbo_actor import TurboMPCActor

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Native actor requires CUDA")
DEVICE = torch.device("cuda:0")


def forbid_condensed_solver(*args, **kwargs):
    raise AssertionError("TurboMPCActor must not call the condensed DifferentiableQP")


@pytest.fixture
def ipc_context():
    """Own a real PathCover/IPC snapshot with compact SFC storage."""
    planner = Planner(Config.load("avoid", 10), [10, 0, 1])
    state, reference, corridors = planner.context(np.array([0, 0, 1, 0.1, 0, 0, 0, 0, 0.0]))
    capacity = max(1, max(len(faces) for faces in corridors))
    planes, mask, schedule = pack_corridors(corridors, capacity)
    assert mask.any(), "The regression must contain actual PathCover constraints"
    values = {
        "policy": np.zeros(124, np.float32),
        "mpc_state": state,
        "reference": reference,
        "planes": planes,
        "plane_mask": mask,
        "corridor_index": schedule,
        "rotation": np.eye(3).reshape(-1),
        "braking": np.array([False]),
        "yaw_target": np.array([0.0]),
        "sfc_valid": np.array([True]),
    }
    obs = TensorDict({key: torch.as_tensor(value, device=DEVICE).unsqueeze(0) for key, value in values.items()}, [1])
    return planner, obs, (state, reference, corridors)


def make_actor(obs):
    actor = TurboMPCActor(obs, {"actor": ["policy"]}, "actor", 4, compile_kernels=False).to(DEVICE)
    actor.solver = forbid_condensed_solver
    return actor


def test_native_actor_forward_uses_native_solver_and_satisfies_ipc_constraints(ipc_context):
    planner, obs, inputs = ipc_context
    actor = make_actor(obs)
    try:
        with torch.no_grad():
            command = actor(obs)
        assert actor.native_solve_calls == 1
        assert actor.last_native_result is not None
        assert actor.last_native_result.status.tolist() == [0]
        assert actor.last_native_result.trajectory.shape == (1, 16, 12)
        assert actor.last_result["solved"].tolist() == [True]
        assert actor.last_result["braking"].tolist() == [False]
        torch.testing.assert_close(
            actor.last_result["solution"],
            actor.last_native_result.controls[:, :-1].flatten(1),
            rtol=0,
            atol=0,
        )
        # Check the independently assembled, published condensed IPC constraints.
        _, _, matrix, lower, upper, state = planner.mpc.problem(*inputs)
        solution = actor.last_result["solution"][0].cpu().numpy()
        products = matrix @ solution
        violation = max(np.maximum(lower - products, 0).max(), np.maximum(products - upper, 0).max())
        assert violation <= 1e-5
        predicted = (planner.mpc.M @ state + planner.mpc.C @ solution).reshape(15, 9)
        np.testing.assert_allclose(actor.last_native_result.states[0, 1:].cpu(), predicted, atol=5e-6, rtol=0)
        expected = flatness(predicted[0, 6:], solution[:3], np.eye(3))
        expected[0] = np.clip(expected[0], 0.08 * 9.81 / 0.7, 0.9 * 9.81 / 0.7)
        np.testing.assert_allclose(command[0].cpu(), expected, atol=2e-6, rtol=0)
    finally:
        actor.close()


def test_native_actor_cost_network_gradient_and_saved_observation_replay(ipc_context, tmp_path):
    planner, obs, _ = ipc_context
    actor = make_actor(obs)
    try:
        command = actor(obs)
        command[:, 2].sum().backward()
        gradients = [parameter.grad for parameter in actor.mlp.parameters() if parameter.requires_grad]
        assert gradients and all(value is not None and torch.isfinite(value).all() for value in gradients)
        assert actor.mlp[-1].bias.grad.abs().max() > 1e-6

        saved = tmp_path / "ipc_context.pt"
        torch.save({key: value.detach().cpu().clone() for key, value in obs.items()}, saved)
        before = command.detach().clone()
        # The replay must not consult mutated observations or the live planner.
        obs["reference"].add_(3)
        obs["planes"].add_(0.25)
        planner.goal[:] = [-5, 2, 2]
        planner.path = np.empty((0, 3))
        replay = TensorDict(torch.load(saved, map_location=DEVICE, weights_only=True), [1])
        with torch.no_grad():
            after = actor(replay)
        assert actor.native_solve_calls == 2
        assert actor.last_result["solved"].all()
        torch.testing.assert_close(after, before, atol=2e-6, rtol=0)
    finally:
        actor.close()


def test_native_actor_mixed_invalid_sfc_skips_solver_and_has_zero_cost_gradient(ipc_context, monkeypatch):
    _, obs, _ = ipc_context
    mixed = torch.cat((obs, obs.clone()), dim=0)
    mixed["sfc_valid"][1] = False
    mixed["mpc_state"][1, 3] = 2.0
    # Invalid geometry/reference must never enter the native problem builder.
    mixed["planes"][1] = torch.nan
    mixed["reference"][1] = torch.nan
    actor = make_actor(mixed)
    outputs = []
    hook = actor.mlp.register_forward_hook(lambda module, inputs, result: outputs.append(result))
    try:
        actions = actor(mixed)
        assert actor.native_solve_calls == 1
        assert actor.last_native_result.status.shape == (1,)
        assert actor.last_result["solved"].tolist() == [True, False]
        assert actor.last_result["braking"].tolist() == [False, True]
        torch.testing.assert_close(actions[1], actor._backup[1], atol=0, rtol=0)
        mixed_grad = torch.autograd.grad(actions[:, 2].sum(), outputs[-1])[0]
        assert torch.isfinite(mixed_grad).all()
        assert mixed_grad[0].abs().max() > 1e-6
        assert torch.equal(mixed_grad[1], torch.zeros_like(mixed_grad[1]))

        isolated = actor(obs)
        isolated_grad = torch.autograd.grad(isolated[:, 2].sum(), outputs[-1])[0]
        torch.testing.assert_close(actions[:1], isolated, atol=2e-6, rtol=0)
        torch.testing.assert_close(mixed_grad[:1], isolated_grad, atol=2e-6, rtol=2e-4)

        calls = actor.native_solve_calls
        monkeypatch.setattr(actor.native_solver, "solve", forbid_condensed_solver)
        invalid = mixed[1:].clone()
        first = actor(invalid, stochastic_output=True)
        assert actor.last_native_result is None
        assert actor.native_solve_calls == calls
        zero_grad = torch.autograd.grad(actor.output_mean.sum(), outputs[-1])[0]
        assert torch.equal(zero_grad, torch.zeros_like(zero_grad))
        second = actor(invalid, stochastic_output=True)
        torch.testing.assert_close(first, second, atol=0, rtol=0)
        torch.testing.assert_close(second, actor._backup, atol=0, rtol=0)
        assert actor.native_solve_calls == calls
        assert actor.get_output_log_prob(second).item() == 0
    finally:
        hook.remove()
        actor.close()


def test_native_actor_close_is_idempotent_and_solver_can_be_explicitly_recreated(ipc_context):
    _, obs, _ = ipc_context
    actor = make_actor(obs)
    try:
        with torch.no_grad():
            before = actor(obs)
        previous = actor.native_solver
        actor.close()
        actor.close()
        assert actor.last_native_result is None

        replacement = actor.native_solver
        assert replacement is not previous
        data, guess = actor.ipc_problem(obs, torch.zeros((1, 7), device=DEVICE))
        with pytest.raises(RuntimeError, match="closed"):
            previous.solve(data, guess)
        with torch.no_grad():
            after = actor(obs)
        assert actor.native_solve_calls == 2
        assert actor.last_result["solved"].all()
        torch.testing.assert_close(after, before, atol=2e-6, rtol=0)
    finally:
        actor.close()
        actor.close()
