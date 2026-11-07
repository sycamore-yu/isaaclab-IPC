"""Observation ownership and replay contracts; no simulated or native CUDA solve."""

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
from tensordict import TensorDict

from isaaclab_ipc.corridor_storage import expand_corridors, pack_corridors
from isaaclab_ipc.frontend import PlanningContext
from isaaclab_ipc.rl.ctbr_actor import CTBRMPCActor
from isaaclab_ipc.tasks.ctbr_env import REFERENCE_DIAGNOSTICS, CTBRMPCEnv, make_ctbr_cfg
from isaaclab_ipc.tasks.env import IPCEnv
from isaaclab_ipc.tasks.mpc_env import IPCMPCEnv
from isaaclab_ipc.turbompc.ctbr import CTBRConfig, CTBRQPBuilder


def forbidden(*args, **kwargs):
    raise AssertionError("Replay must use frozen observations without online planning")


@pytest.fixture
def environment(monkeypatch):
    env = object.__new__(CTBRMPCEnv)
    env.cfg = make_ctbr_cfg()
    env.ctbr_config = CTBRConfig()
    env.sim = SimpleNamespace(device="cpu")
    env.scene = SimpleNamespace(num_envs=1, env_origins=torch.tensor([[1.0, 3.0, 3.0]], dtype=torch.float32))
    pose = torch.tensor([[2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)
    twist = torch.tensor([[0.2, 0.1, 0.0, 0.0, 0.0, 0.03]], dtype=torch.float32)
    env.body_state = lambda: (pose, twist)
    physical = torch.cat(
        (pose[:, :3] - env.scene.env_origins, twist[:, :3], torch.zeros((1, 3)), torch.eye(3).reshape(1, 9)), -1
    )
    monkeypatch.setattr(IPCEnv, "_get_observations", lambda self: {"policy": physical.clone()})
    env.elapsed = torch.tensor([0.01])  # No new LiDAR cloud on this control tick.
    env.frontend_times = []
    env.goal = torch.tensor([10.0, 0.0, 1.0])
    old = np.array(
        [[1.0, 0, 0, -5], [-1, 0, 0, 0], [0, 1, 0, -5], [0, -1, 0, -5], [0, 0, 1, -3], [0, 0, -1, 0], [1, 0, 0, -5]]
    )
    new = old.copy()
    new[[0, -1], 3] = -8
    reference = np.zeros((15, 9))
    reference[:, :3] = np.column_stack((1 + 0.2 * np.arange(15), np.zeros(15), np.ones(15)))
    reference[:, 3] = 2
    context = PlanningContext(physical[0, :9].numpy().copy(), reference, [old] * 7 + [new] * 8, True)
    path = np.column_stack((1 + 0.02 * np.arange(160), np.zeros(160), np.ones(160)))
    planner = SimpleNamespace(
        cfg=SimpleNamespace(params={"yaw_ctrl_flag": False}),
        goal=np.array([10, 0, 1]),
        braking=False,
        path=path,
        nearest=0,
    )
    env.planners = [planner]
    env.planning_batch = SimpleNamespace(contexts=lambda *args: [context])
    return SimpleNamespace(
        env=env, context=context, pose=pose, twist=twist, physical=physical, old=old, new=new, planner=planner
    )


def install_adaptation(monkeypatch, data, *, rejected=False, corrupt=None):
    calls = []
    updated_reference = data.context.reference.copy()
    updated_reference[:, 0] = 1.0 + 0.08 * np.arange(15)
    updated_reference[:, 3] = 0.8
    updated_corridors = [data.old] * 3 + [data.new] * 12
    diagnostics = dict(
        version=1,
        applied=True,
        reason_code=9,
        actual_backup_certified=not rejected,
        nominal_backup_certified=False,
        selected_backup_certified=not rejected,
        reference_scale=0.4,
        nominal_speed=2.0,
        selected_speed=0.8,
        actual_backup_steps=20,
        selected_backup_steps=30,
        containing_corridor=0,
    )

    def adapt(context, state, config, *, path, nearest):
        calls.append(state.copy())
        assert context is data.context and config is data.env.ctbr_config
        assert nearest == 0 and not path.flags.writeable and not state.flags.writeable
        np.testing.assert_array_equal(path, data.planner.path)
        assert not np.shares_memory(path, data.planner.path)
        expected = np.r_[data.physical[0, :3].numpy(), 1.0, 0.0, 0.0, 0.0, data.physical[0, 3:6].numpy()]
        np.testing.assert_array_equal(state, expected)
        changed = PlanningContext(context.state, updated_reference.copy(), updated_corridors, context.valid)
        if corrupt == "state":
            changed.state = context.state + 1
        elif corrupt == "valid":
            changed.valid = not context.valid
        return changed, diagnostics.copy()

    monkeypatch.setattr("isaaclab_ipc.tasks.ctbr_env.adapt_ctbr_reference", adapt)
    return calls, updated_reference, updated_corridors, diagnostics


def test_default_ipc_hook_is_identity_and_retains_original_observation(environment, monkeypatch):
    data = environment
    contexts = [data.context]
    assert IPCMPCEnv._adapt_planning_contexts(data.env, contexts) is contexts
    monkeypatch.setattr(
        data.env, "_adapt_planning_contexts", lambda value: IPCMPCEnv._adapt_planning_contexts(data.env, value)
    )
    result = IPCMPCEnv._get_observations(data.env)
    np.testing.assert_array_equal(result["reference"][0], data.context.reference)
    assert result["corridor_index"].tolist() == [[0] * 7 + [1] * 8]
    assert result["policy"].shape == (1, 124)
    assert "nominal_reference" not in result and "ctbr_reference_diagnostics" not in result


def test_adaptation_precedes_packing_features_and_freezes_both_contexts(environment, monkeypatch):
    data = environment
    calls, reference, corridors, diagnostics = install_adaptation(monkeypatch, data)
    original_state = data.context.state.copy()
    original_reference = data.context.reference.copy()
    nominal_planes, nominal_mask, nominal_index = pack_corridors(data.context.corridors, data.env.cfg.max_planes)
    observation = data.env._get_observations()
    assert len(calls) == 1 and not hasattr(data.env, "_ctbr_reference_capture")
    np.testing.assert_array_equal(data.context.state, original_state)
    np.testing.assert_array_equal(data.context.reference, original_reference)
    np.testing.assert_array_equal(observation["reference"][0], reference)
    np.testing.assert_array_equal(observation["ctbr_reference"][0, :, :3], reference[:, :3])
    np.testing.assert_array_equal(observation["ctbr_reference"][0, :, 7:], reference[:, 3:6])
    assert observation["corridor_index"].tolist() == [[0] * 3 + [1] * 12]
    np.testing.assert_array_equal(observation["nominal_reference"][0], original_reference)
    np.testing.assert_array_equal(observation["nominal_planes"][0], nominal_planes)
    np.testing.assert_array_equal(observation["nominal_plane_mask"][0], nominal_mask)
    np.testing.assert_array_equal(observation["nominal_corridor_index"][0], nominal_index)
    np.testing.assert_array_equal(
        observation["ctbr_reference_diagnostics"][0], [diagnostics[k] for k in REFERENCE_DIAGNOSTICS]
    )
    policy = observation["policy"][0]
    torch.testing.assert_close(
        policy[18:63], torch.from_numpy(((reference[:, :3] - original_state[:3]) / 10).ravel()).float()
    )
    torch.testing.assert_close(policy[63:108], torch.from_numpy((reference[:, 3:6] / 10).ravel()).float())
    expected_margin = np.array(
        [
            min(-(faces[:, :3] @ point + faces[:, 3]) / np.linalg.norm(faces[:, :3], axis=1))
            for point, faces in zip(reference[:, :3], corridors, strict=True)
        ]
    )
    torch.testing.assert_close(policy[108:123], torch.from_numpy(np.clip(expected_margin, -10, 10) / 10).float())
    dense, mask = expand_corridors(observation)
    for stage, faces in enumerate(corridors):
        np.testing.assert_array_equal(dense[0, stage][mask[0, stage]], faces[::-1])
    frozen = {key: value.clone() for key, value in observation.items()}
    data.pose.add_(2)
    data.twist.add_(3)
    data.context.reference += 4
    data.old += 5
    data.planner.path += 6
    for key, value in frozen.items():
        torch.testing.assert_close(observation[key], value, atol=0, rtol=0)


def test_uncertified_backup_does_not_redefine_geometric_validity(environment, monkeypatch):
    install_adaptation(monkeypatch, environment, rejected=True)
    result = environment.env._get_observations()
    assert result["sfc_valid"].tolist() == [[True]]
    field = REFERENCE_DIAGNOSTICS.index("actual_backup_certified")
    assert result["ctbr_reference_diagnostics"][0, field] == 0
    assert result["plane_mask"].all(-1).tolist() == [[False, False]]  # Padding remains masked.
    assert (result["corridor_index"] >= 0).all()


@pytest.mark.parametrize("corrupt", ["state", "valid"])
def test_adapter_cannot_replace_actual_state_or_geometry_validity(environment, monkeypatch, corrupt):
    install_adaptation(monkeypatch, environment, corrupt=corrupt)
    with pytest.raises(RuntimeError, match="preserve measured state and geometry validity"):
        environment.env._get_observations()
    assert not hasattr(environment.env, "_ctbr_reference_capture")


def test_frozen_context_replays_builder_actor_and_checkpoint_without_frontend(environment, monkeypatch):
    data = environment
    calls, _, _, _ = install_adaptation(monkeypatch, data)
    frozen = TensorDict(data.env._get_observations(), [1]).clone()

    def configure_base(env):
        env.single_observation_space = gym.spaces.Dict(
            {"policy": gym.spaces.Box(-np.inf, np.inf, shape=(127,), dtype=np.float32)}
        )

    monkeypatch.setattr(IPCEnv, "_configure_gym_env_spaces", configure_base)
    data.env._configure_gym_env_spaces()
    for name, value in frozen.items():
        assert data.env.single_observation_space[name].contains(value[0].numpy())
    monkeypatch.setattr(data.env.planning_batch, "contexts", forbidden)
    monkeypatch.setattr("isaaclab_ipc.tasks.ctbr_env.adapt_ctbr_reference", forbidden)
    data.context.reference[:] = 100
    data.old[:] = 100
    data.planner.path[:] = 100
    builder = CTBRQPBuilder()
    costs = torch.zeros((1, 7), dtype=torch.float64, requires_grad=True)
    first, first_guess = builder(frozen, costs)
    second, second_guess = builder(frozen.clone(), costs)
    for name in ("initial_state", "D", "E", "q", "G", "lower", "upper"):
        torch.testing.assert_close(getattr(first, name), getattr(second, name), atol=0, rtol=0)
    torch.testing.assert_close(first_guess, second_guess, atol=0, rtol=0)

    def synthetic_solve(inputs, guess):
        # Differentiable substitute tests route/replay only, not numerical MPC.
        signal = 1e-6 * inputs.q[:, 1:, :3].sum((1, 2))[:, None, None]
        controls = guess[..., 10:] + signal
        return SimpleNamespace(
            status=torch.zeros(1, dtype=torch.int32),
            controls=controls,
            states=guess[..., :10],
            trajectory=torch.cat((guess[..., :10], controls), -1),
            errors=torch.zeros((1, 3)),
            normal_cone_error=torch.zeros(1),
            admm=SimpleNamespace(status=torch.zeros(1, dtype=torch.int32), iterations=torch.ones(1, dtype=torch.int32)),
        )

    actors = [CTBRMPCActor(frozen, {"actor": ["policy"]}, "actor", 4, compile_kernels=False) for _ in range(2)]
    for actor in actors:
        actor._native_solver = SimpleNamespace(solve=synthetic_solve, close=lambda: None)
    try:
        expected = actors[0](frozen)
        actors[1].load_state_dict(actors[0].state_dict())
        actual = actors[1](frozen.clone())
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        gradients = torch.autograd.grad(actors[1].output_mean.sum(), actors[1].mlp.parameters())
        assert all(torch.isfinite(value).all() for value in gradients)
        assert any(torch.count_nonzero(value) for value in gradients)
        assert len(calls) == 1
    finally:
        for actor in actors:
            actor.close()


def test_real_adapter_keeps_double_state_with_float32_translated_context(environment):
    """Isaac's float32 local position and CTBR's double subtraction can differ."""
    data = environment
    data.pose[0, 0] = 8.8765
    data.env.scene.env_origins[0, 0] = -1.2345
    expected_position = data.pose[0, :3].double() - data.env.scene.env_origins[0].double()
    local_position = data.pose[0, :3] - data.env.scene.env_origins[0]
    shift = float(local_position[0] - data.physical[0, 0])
    data.physical[0, :3].copy_(local_position)
    data.context.state[:3] = local_position.numpy()
    data.context.reference[:, 0] += shift
    data.planner.path[:, 0] += shift
    for faces in (data.old, data.new):
        faces[:, 3] -= faces[:, 0] * shift
    before_state = data.context.state.copy()
    before_reference = data.context.reference.copy()
    before_planes, before_mask, before_index = pack_corridors(data.context.corridors, data.env.cfg.max_planes)
    assert not torch.equal(expected_position, local_position.double())
    torch.testing.assert_close(expected_position.float(), local_position, atol=0, rtol=0)

    result = data.env._get_observations()  # Exercise the real pure adapter and env hook together.

    torch.testing.assert_close(result["ctbr_state"][0, :3], expected_position, atol=0, rtol=0)
    torch.testing.assert_close(result["mpc_state"][0, :3], local_position.double(), atol=0, rtol=0)
    np.testing.assert_array_equal(data.context.state, before_state)
    np.testing.assert_array_equal(data.context.reference, before_reference)
    np.testing.assert_array_equal(result["nominal_reference"][0], before_reference)
    np.testing.assert_array_equal(result["nominal_planes"][0], before_planes)
    np.testing.assert_array_equal(result["nominal_plane_mask"][0], before_mask)
    np.testing.assert_array_equal(result["nominal_corridor_index"][0], before_index)
    assert result["sfc_valid"].tolist() == [[True]]
    assert torch.isfinite(result["ctbr_reference_diagnostics"]).all()
    assert (result["corridor_index"] >= 0).all()
    dense, masks = expand_corridors(result)
    for stage in range(15):
        faces = dense[0, stage][masks[0, stage]].numpy()
        assert any(np.array_equal(faces, original[::-1]) for original in (data.old, data.new))
