"""CTBR Actor/task contracts on CPU and actual native CUDA integration.

GPU cases use a fixed real PathCover context with a 0.4 m goal. They are
interface/gradient regressions; independent CTBR finite differences and Newton
closed-loop acceptance live in separate harnesses.
"""

import copy
import math
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
from tensordict import TensorDict

from isaaclab_ipc.corridor_storage import pack_corridors
from isaaclab_ipc.mpc import Config
from isaaclab_ipc.planner import Planner
from isaaclab_ipc.rl.actor import MPCActor
from isaaclab_ipc.rl.ctbr_actor import CTBRMPCActor
from isaaclab_ipc.tasks.ctbr_env import CTBRMPCEnv, checked_articulation_mass, make_ctbr_cfg
from isaaclab_ipc.tasks.env import IPCEnv
from isaaclab_ipc.tasks.mpc_env import IPCMPCEnv
from isaaclab_ipc.turbompc.ctbr import CTBRConfig
from isaaclab_ipc.turbompc.quadrotor import CTBRDynamics, reference_from_ipc


def forbid_solver(*args, **kwargs):
    raise AssertionError("This path must not invoke an excluded solver")


def fixed_context(device="cpu"):
    """Capture an actual compact PathCover schedule; never select cases by result."""
    planner = Planner(Config.load("avoid", 10, profile="real"), [0.4, 0, 1])
    state, reference, corridors = planner.context(np.array([0, 0, 1, 0, 0, 0, 0, 0, 0.0]))
    capacity = max(1, max(len(faces) for faces in corridors))
    planes, mask, schedule = pack_corridors(corridors, capacity)
    assert mask.any(), "The fixed regression must include actual PathCover faces"
    values = {
        "policy": np.zeros(127, np.float32),
        "mpc_state": state,
        "reference": reference,
        "planes": planes,
        "plane_mask": mask,
        "corridor_index": schedule,
        "rotation": np.eye(3).reshape(-1),
        "braking": np.array([False]),
        "yaw_target": np.array([0.0]),
        "sfc_valid": np.array([True]),
        "ctbr_state": np.array([0, 0, 1, 1, 0, 0, 0, 0, 0, 0.0]),
        "body_rates": np.zeros(3),
    }
    obs = TensorDict({key: torch.as_tensor(value, device=device)[None] for key, value in values.items()}, [1])
    obs["ctbr_reference"] = reference_from_ipc(obs["reference"], obs["yaw_target"])
    return planner, obs


def make_actor(obs):
    actor = CTBRMPCActor(obs, {"actor": ["policy"]}, "actor", 4, compile_kernels=False).to(
        obs.device or obs["policy"].device
    )
    return actor


def action_cost_jacobian(actions, costs):
    """Check every physical action; a bound-active component can have zero VJP."""
    return torch.stack(
        [torch.autograd.grad(actions[:, coordinate].sum(), costs, retain_graph=True)[0] for coordinate in range(4)],
        dim=1,
    )


def bare_environment(num_envs=1):
    """Exercise real task methods without creating a Newton simulation."""
    env = object.__new__(CTBRMPCEnv)
    env.ctbr_config = CTBRConfig()
    env.cfg = make_ctbr_cfg()
    env.scene = SimpleNamespace(num_envs=num_envs, env_origins=torch.zeros((num_envs, 3), dtype=torch.float64))
    env._pending = None
    return env


def mass_robot(link_masses):
    return SimpleNamespace(
        num_instances=link_masses.shape[0],
        num_bodies=5,
        body_names=["base", "rotor_0", "rotor_1", "rotor_2", "rotor_3"],
        data=SimpleNamespace(body_mass=SimpleNamespace(torch=link_masses)),
    )


def test_articulation_mass_includes_all_four_fixed_rotor_links():
    link_masses = torch.tensor([[0.746, 0.001, 0.001, 0.001, 0.001]] * 2, requires_grad=True)
    before = link_masses.detach().clone()
    masses = checked_articulation_mass(mass_robot(link_masses), 0.75)
    torch.testing.assert_close(masses, torch.full((2,), 0.75), atol=1e-6, rtol=0)
    torch.testing.assert_close(link_masses, before, atol=0, rtol=0)
    assert not masses.requires_grad


@pytest.mark.parametrize("second_base_mass", [0.756, float("nan")])
def test_articulation_mass_rejects_wrong_or_nonfinite_mass_per_instance(second_base_mass):
    link_masses = torch.tensor(
        [[0.746, 0.001, 0.001, 0.001, 0.001], [second_base_mass, 0.001, 0.001, 0.001, 0.001]]
    )
    with pytest.raises(ValueError, match=r"actual=.*expected=0\.75 kg"):
        checked_articulation_mass(mass_robot(link_masses), 0.75)


def test_articulation_mass_rejects_incomplete_body_axis():
    with pytest.raises(ValueError, match=r"actual=\(2, 1\), expected=\(2, 5\)"):
        checked_articulation_mass(mass_robot(torch.full((2, 1), 0.75)), 0.75)


def test_environment_closes_after_articulation_mass_check_fails(monkeypatch):
    closed = []

    def fake_initialize(env, cfg, render_mode=None, **kwargs):
        env.robot = mass_robot(torch.tensor([[0.756, 0.001, 0.001, 0.001, 0.001]]))

    monkeypatch.setattr(IPCMPCEnv, "__init__", fake_initialize)
    monkeypatch.setattr(CTBRMPCEnv, "close", lambda env: closed.append(env))
    with pytest.raises(ValueError, match=r"actual=.*expected=0\.75 kg"):
        CTBRMPCEnv(make_ctbr_cfg())
    assert len(closed) == 1


def test_all_invalid_context_skips_both_solvers_and_has_zero_cost_gradient(monkeypatch):
    _, obs = fixed_context()
    obs["sfc_valid"].fill_(False)
    obs["mpc_state"][:, 3] = 1.0
    obs["ctbr_state"][:, 7] = 1.0
    obs["ctbr_reference"].fill_(torch.nan)
    obs["planes"].fill_(torch.nan)
    actor = make_actor(obs)
    monkeypatch.setattr("isaaclab_ipc.rl.ctbr_actor.TorchMPC", forbid_solver)
    outputs = []
    hook = actor.mlp.register_forward_hook(lambda module, inputs, result: outputs.append(result))
    try:
        command = actor(obs)
        assert actor.obs_dim == 127
        assert actor.native_solve_calls == 0 and actor._native_solver is None
        assert actor.last_native_result is None
        assert actor.last_result["solved"].tolist() == [False]
        assert actor.last_result["braking"].tolist() == [True]
        assert actor.last_result["model"] == "ctbr"
        assert actor.last_result["solution"].shape == (1, 60)
        assert actor.last_result["trajectory"].shape == (1, 16, 14)
        assert torch.isfinite(command).all()
        torch.testing.assert_close(command, actor._backup, atol=0, rtol=0)
        assert (command.double() >= actor.ctbr_problem.action_lower - 1e-6).all()
        assert (command.double() <= actor.ctbr_problem.action_upper + 1e-6).all()
        gradient = torch.autograd.grad(actor.output_mean.sum(), outputs[-1])[0]
        assert torch.equal(gradient, torch.zeros_like(gradient))
        first, second = actor(obs, stochastic_output=True), actor(obs, stochastic_output=True)
        torch.testing.assert_close(first, second, atol=0, rtol=0)
        assert actor.get_output_log_prob(second).item() == 0
        assert actor.native_solve_calls == 0
    finally:
        hook.remove()
        actor.close()
        actor.close()


def test_sqp_success_with_exhausted_inner_qp_routes_to_recovery_and_zero_gradient():
    """Synthetic status routing only; CUDA tests independently cover real solves."""
    _, obs = fixed_context()
    actor = make_actor(obs)
    outputs = []
    hook = actor.mlp.register_forward_hook(lambda module, inputs, result: outputs.append(result))

    def exhausted_qp(inputs, guess):
        # Keep a differentiable candidate so the zero recovery gradient is tested.
        influence = 1e-5 * inputs.q[:, 1:, :3].sum((1, 2))[:, None, None]
        controls = guess[..., 10:] + influence
        return SimpleNamespace(
            status=torch.zeros(1, dtype=torch.int32),
            controls=controls,
            states=guess[..., :10],
            trajectory=torch.cat((guess[..., :10], controls), -1),
            errors=torch.zeros((1, 3), dtype=torch.float64),
            normal_cone_error=torch.zeros(1, dtype=torch.float64),
            admm=SimpleNamespace(status=torch.ones(1, dtype=torch.int32), iterations=torch.tensor([16384])),
        )

    actor._native_solver = SimpleNamespace(solve=exhausted_qp, close=lambda: None)
    try:
        action = actor(obs)
        assert actor.native_solve_calls == 1
        assert actor.last_native_result.status.tolist() == [0]
        assert actor.last_result["raw_qp_status"].tolist() == [1]
        assert actor.last_result["solved"].tolist() == [False]
        assert actor.last_result["braking"].tolist() == [True]
        torch.testing.assert_close(action, actor._backup, atol=0, rtol=0)
        gradient = torch.autograd.grad(actor.output_mean.sum(), outputs[-1])[0]
        assert torch.equal(gradient, torch.zeros_like(gradient))
        assert actor.get_output_log_prob(action).item() == 0
    finally:
        hook.remove()
        actor.close()


def test_checkpoint_rejects_old_jerk_semantics_and_changed_physics(tmp_path):
    _, obs = fixed_context()
    actor, restored = make_actor(obs), make_actor(obs)
    old = MPCActor(obs, {"actor": ["policy"]}, "actor", 4)
    try:
        path = tmp_path / "ctbr_actor.pt"
        torch.save(actor.state_dict(), path)
        saved = torch.load(path, map_location="cpu", weights_only=True)
        restored.load_state_dict(saved)
        assert saved["_extra_state"]["policy_features"] == 127
        assert saved["ctbr_problem._extra_state"]["config"]["mass"] == 0.75
        # Equal seven-head and 127-input widths must not make a jerk state valid.
        with pytest.raises(RuntimeError, match="Missing key"):
            restored.load_state_dict(old.state_dict())
        wrong = copy.deepcopy(saved)
        wrong["_extra_state"]["model"]["mass"] = 0.752
        with pytest.raises(ValueError, match="checkpoint"):
            restored.load_state_dict(wrong)
        wrong = copy.deepcopy(saved)
        wrong["ctbr_problem._extra_state"]["cost_semantics_version"] = 0
        with pytest.raises(RuntimeError, match="cost semantics"):
            restored.load_state_dict(wrong)
    finally:
        actor.close()
        restored.close()


def test_task_factory_registration_and_dynamic_observation_schema(monkeypatch):
    from isaaclab_ipc.rl.config import CTBRTrainCfg

    cfg = make_ctbr_cfg()
    assert cfg.observation_space == 127 and cfg.reference_strides == (10,)
    assert CTBRTrainCfg().actor.class_name == "isaaclab_ipc.rl.ctbr_actor:CTBRMPCActor"
    for scenario in ("Avoid", "Forest"):
        spec = gym.spec(f"Isaac-IPC-{scenario}-ACMPC-CTBR-Direct")
        assert spec.entry_point == "isaaclab_ipc.tasks.ctbr_env:CTBRMPCEnv"
        assert spec.kwargs["rsl_rl_cfg_entry_point"] == "isaaclab_ipc.rl.config:CTBRTrainCfg"
        suffix = "make_forest_ctbr_cfg" if scenario == "Forest" else "make_ctbr_cfg"
        assert spec.kwargs["env_cfg_entry_point"].endswith(suffix)

    def configure_base(self):
        self.single_observation_space = gym.spaces.Dict(
            {"policy": gym.spaces.Box(-np.inf, np.inf, shape=(self.cfg.observation_space,), dtype=np.float32)}
        )

    monkeypatch.setattr(IPCEnv, "_configure_gym_env_spaces", configure_base)
    env = bare_environment(2)
    for capacity in (4, 69):
        env.cfg.max_planes = capacity
        env._configure_gym_env_spaces()
        single = env.single_observation_space
        assert single["policy"].shape == (127,)
        assert single["ctbr_state"].shape == (10,)
        assert single["ctbr_reference"].shape == (15, 10)
        assert single["body_rates"].shape == (3,)
        assert single["planes"].shape == (2, capacity, 4)
        assert single["plane_mask"].shape == (2, capacity)
        assert env.observation_space["planes"].shape == (2, 2, capacity, 4)
        assert single["ctbr_state"].dtype == np.float64


def test_environment_physical_snapshot_preserves_axes_and_owns_tensors(monkeypatch):
    env = bare_environment(2)
    half = math.sqrt(0.5)
    pose = torch.tensor(
        [[5.0, 6.0, 7.0, 0.0, 0.0, half, half], [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]], dtype=torch.float64
    )
    twist = torch.tensor([[1.0, 2.0, 3.0, 0.0, 2.0, 0.3], [-0.1, 0.2, -0.3, 0.4, -0.5, 0.6]], dtype=torch.float64)
    env.scene.env_origins = torch.tensor([[4.0, 4.0, 4.0], [0.0, 1.0, 2.0]], dtype=torch.float64)
    env.body_state = lambda: (pose, twist)
    reference = torch.zeros((2, 15, 9), dtype=torch.float64)
    reference[..., :3] = torch.tensor([0.2, -0.1, 1.0])
    reference[..., 3:6] = torch.tensor([0.3, 0.1, -0.2])
    base = {
        "policy": torch.arange(248, dtype=torch.float32).reshape(2, 124),
        "reference": reference,
        "yaw_target": torch.tensor([[0.2], [-0.3]], dtype=torch.float64),
    }
    monkeypatch.setattr(
        IPCMPCEnv, "_get_observations", lambda self: {key: value.clone() for key, value in base.items()}
    )
    obs = env._get_observations()
    expected = torch.tensor(
        [[1.0, 2.0, 3.0, half, 0.0, 0.0, half, 1.0, 2.0, 3.0], [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, -0.1, 0.2, -0.3]],
        dtype=torch.float64,
    )
    rates = torch.tensor([[2.0, 0.0, 0.3], [0.4, -0.5, 0.6]], dtype=torch.float64)
    torch.testing.assert_close(obs["ctbr_state"], expected, atol=0, rtol=0)
    torch.testing.assert_close(obs["body_rates"], rates, atol=1e-15, rtol=0)
    torch.testing.assert_close(obs["policy"][:, :124], base["policy"], atol=0, rtol=0)
    torch.testing.assert_close(obs["policy"][:, 124:], (rates / torch.tensor([1.5, 1.5, 0.1])).float())
    torch.testing.assert_close(obs["ctbr_reference"][..., 7:], reference[..., 3:6])
    assert env.current_context is obs
    frozen = {name: obs[name].clone() for name in ("ctbr_state", "body_rates", "ctbr_reference", "policy")}
    pose.add_(1)
    twist.add_(2)
    reference.add_(3)
    for name, value in frozen.items():
        torch.testing.assert_close(obs[name], value, atol=0, rtol=0)


def result_for_state(state, mean_specific_thrust):
    model = CTBRDynamics()
    z = torch.from_numpy(np.r_[state, 0.75 * mean_specific_thrust, 0.0, 0.0, 0.0])
    return {
        "solved": np.array([True]),
        "sfc_valid": np.array([True]),
        "braking": np.array([False]),
        "state": state[None].copy(),
        "navigation_state": np.r_[state[:3], state[7:], 0.0, 0.0, 0.0][None],
        "solution": np.zeros((1, 60)),
        "predicted": model(z).numpy()[None],
        "predicted_acceleration": model.acceleration(z).numpy()[None],
    }


@pytest.mark.parametrize("kind", ["velocity", "acceleration"])
def test_consumer_checks_actual_sampled_action_and_never_parses_plan_as_jerk(monkeypatch, kind):
    env = bare_environment()
    planner = Planner(Config.load("avoid", 10, profile="real"), [0.4, 0, 1])
    planner.previous = np.arange(45).reshape(15, 3).copy()
    old_plan = planner.previous.copy()
    monkeypatch.setattr(planner, "consume_state", forbid_solver)
    pitch = np.pi / 2 if kind == "velocity" else np.arccos(-0.02)
    state = np.array([0.0, 0.0, 1.0, np.cos(pitch / 2), 0.0, np.sin(pitch / 2), 0.0, 0.0, 0.0, 0.0])
    if kind == "velocity":
        state[7] = 9.5
    mean = 3.0 if kind == "velocity" else 5.0
    data = result_for_state(state, mean)
    assert not env._consume_control_result(planner, data, 0, 0.0, np.array([mean, 0.0, 0.0, 0.0]))
    assert env._consume_control_result(planner, data, 0, 0.0, np.array([12.0, 0.0, 0.0, 0.0]))
    np.testing.assert_array_equal(planner.previous, old_plan)
    assert planner.failures == 0
    data["solved"][:] = False
    data["braking"][:] = True
    env._consume_control_result(planner, data, 0, 0.01, np.array([mean, 0.0, 0.0, 0.0]))
    assert planner.failures == 1 and planner.recovery_events == 1 and planner.brake_steps == 1
    np.testing.assert_array_equal(planner.previous, old_plan)


def test_environment_rejects_jerk_result_schema_and_double_record():
    env = bare_environment()
    valid = {"model": "ctbr", "state": torch.zeros((1, 10)), "solution": torch.zeros((1, 60))}
    for invalid in (
        {**valid, "model": "ipc"},
        {**valid, "state": torch.zeros((1, 9))},
        {**valid, "solution": torch.zeros((1, 45))},
    ):
        with pytest.raises(ValueError, match="CTBR result schema"):
            env.record_mpc_result(invalid)
    env.record_mpc_result(valid)
    assert env._pending is valid
    with pytest.raises(RuntimeError, match="Previous MPC result"):
        env.record_mpc_result(valid)


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="Actual native CTBR Actor requires CUDA")


@requires_cuda
def test_cuda_native_ctbr_forward_gradient_disk_replay_and_close(tmp_path):
    torch.manual_seed(817)
    planner, obs = fixed_context("cuda:0")
    actor = make_actor(obs)
    outputs = []
    hook = actor.mlp.register_forward_hook(lambda module, inputs, result: outputs.append(result))
    try:
        command = actor(obs)
        hook.remove()
        result = actor.last_native_result
        assert actor.native_solve_calls == 1
        assert result.status.tolist() == [0] and result.admm.status.tolist() == [0]
        assert result.trajectory.shape == (1, 16, 14) and result.controls.shape == (1, 16, 4)
        assert actor.last_result["solved"].tolist() == [True] and not actor.last_result["braking"].any()
        expected = result.controls[:, 0].detach().clone()
        expected[:, 0] /= 0.75
        torch.testing.assert_close(command.double(), expected, atol=5e-7, rtol=1e-7)
        torch.testing.assert_close(actor.last_result["solution"], result.controls[:, :-1].flatten(1), atol=0, rtol=0)
        data, _ = actor.ctbr_problem(obs, actor.mlp(actor.get_latent(obs)))
        product = (data.G @ result.trajectory.detach()[..., None]).squeeze(-1)
        violation = torch.maximum(data.lower - product, product - data.upper).clamp_min(0)
        assert float(violation.max()) <= 1e-5
        assert float(actor.ctbr_problem.path_constraints().violations(result.trajectory.detach()).max()) <= 1e-5
        jacobian = action_cost_jacobian(command, outputs[-1])
        assert torch.isfinite(jacobian).all() and jacobian.abs().max() > 1e-7
        # This fixed PathCover context saturates the first pitch-rate input.
        # Its zero sensitivity is correct and must not mask other learnable axes.
        torch.testing.assert_close(command[:, 2].double(), actor.ctbr_problem.action_upper[2:3], atol=1e-6, rtol=0)
        assert jacobian[:, 2].abs().max() < 1e-7
        command.sum().backward()
        gradients = [parameter.grad for parameter in actor.mlp.parameters() if parameter.requires_grad]
        assert gradients and all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
        assert actor.mlp[-1].bias.grad.abs().max() > 1e-7
        saved = tmp_path / "ctbr_observation.pt"
        torch.save({key: value.detach().cpu().clone() for key, value in obs.items()}, saved)
        before = command.detach().clone()
        obs["ctbr_reference"].add_(0.3)
        obs["planes"].add_(0.25)
        planner.goal[:] = [-5.0, 2.0, 2.0]
        planner.path = np.empty((0, 3))
        replay = TensorDict(torch.load(saved, map_location="cuda:0", weights_only=True), [1])
        with torch.no_grad():
            after = actor(replay)
        assert actor.native_solve_calls == 2 and actor.last_result["solved"].all()
        torch.testing.assert_close(after, before, atol=2e-6, rtol=0)
        previous = actor.native_solver
        actor.close()
        actor.close()
        assert actor.last_native_result is None
        with pytest.raises(RuntimeError, match="closed"):
            previous.solve(data)
        with torch.no_grad():
            reopened = actor(replay)
        assert actor.native_solver is not previous and actor.native_solve_calls == 3
        torch.testing.assert_close(reopened, before, atol=2e-6, rtol=0)
    finally:
        hook.remove()
        actor.close()


@requires_cuda
def test_cuda_mixed_invalid_ctbr_context_isolated_in_forward_and_backward():
    torch.manual_seed(817)
    _, obs = fixed_context("cuda:0")
    mixed = torch.cat((obs, obs.clone()), 0)
    mixed["sfc_valid"][1] = False
    mixed["planes"][1] = torch.nan
    mixed["ctbr_reference"][1] = torch.nan
    mixed["mpc_state"][1, 3] = 2.0
    mixed["ctbr_state"][1, 7] = 2.0
    actor = make_actor(mixed)
    outputs = []
    hook = actor.mlp.register_forward_hook(lambda module, inputs, result: outputs.append(result))
    try:
        actions = actor(mixed)
        assert actor.native_solve_calls == 1 and actor.last_native_result.status.tolist() == [0]
        assert actor.last_result["solved"].tolist() == [True, False]
        assert actor.last_result["braking"].tolist() == [False, True]
        torch.testing.assert_close(actions[1], actor._backup[1], atol=0, rtol=0)
        mixed_gradient = action_cost_jacobian(actions, outputs[-1])
        assert torch.isfinite(mixed_gradient).all() and mixed_gradient[0].abs().max() > 1e-7
        assert torch.equal(mixed_gradient[1], torch.zeros_like(mixed_gradient[1]))
        isolated = actor(obs)
        isolated_gradient = action_cost_jacobian(isolated, outputs[-1])
        assert actor.native_solve_calls == 2
        torch.testing.assert_close(actions[:1], isolated, atol=2e-6, rtol=0)
        torch.testing.assert_close(mixed_gradient[:1], isolated_gradient, atol=2e-6, rtol=2e-4)
    finally:
        hook.remove()
        actor.close()
