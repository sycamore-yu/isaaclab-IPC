"""CPU checks for opt-in training phase observation and mutation rejection."""

import hashlib
import importlib.util
import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from types import SimpleNamespace

import pytest
import torch
from rsl_rl.utils.logger import Logger

from isaaclab_ipc.tasks.rewards import NavigationRewardCfg

_spec = importlib.util.spec_from_file_location("acmpc_cli_audit", Path(__file__).parents[1] / "scripts/acmpc.py")
_cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cli)
TrainingAudit = _cli.TrainingAudit


def observation(batch):
    return {"policy": torch.zeros(batch, 127)}


class Actor(torch.nn.Module):
    def forward(self, obs):
        return obs["policy"][:, :4] + 1


class Planner:
    def __init__(self):
        self.calls = dict(context=0, update_cloud=0, consume_recovery=0)
        self.lock = Lock()

    def context(self, value):
        with self.lock:
            self.calls["context"] += 1
        return value + 17

    def update_cloud(self, cloud, position):
        with self.lock:
            self.calls["update_cloud"] += 1
        return cloud, position

    def consume_recovery(self, state, solved):
        with self.lock:
            self.calls["consume_recovery"] += 1
        return not solved


class World:
    def __init__(self):
        self.planner = Planner()
        self.records = []
        self.resets = 0

    def record_mpc_result(self, result):
        self.records.append(result)

    def _reset_idx(self, ids):
        self.resets += 1
        self.planner = Planner()
        return ids


class Environment:
    def __init__(self):
        self.unwrapped = World()
        self.steps = 0

    def step(self, action):
        self.steps += 1
        planner = self.unwrapped.planner
        planner.consume_recovery(0, True)
        planner.update_cloud([], [0, 0, 0])
        planner.context(self.steps)
        return action


class Algorithm:
    def __init__(self, actor):
        self.actor = actor
        self.update_calls = 0

    def update(self):
        self.update_calls += 1
        return [self.actor(observation(8)) for _ in range(2)]


@pytest.fixture
def setup(tmp_path):
    actor = Actor()
    env = Environment()
    algorithm = Algorithm(actor)
    output = tmp_path / "training-audit.json"
    audit = TrainingAudit(algorithm, env, actor, Planner, output)
    return SimpleNamespace(actor=actor, env=env, algorithm=algorithm, output=output, audit=audit)


def test_actual_phase_counts_and_new_planners_are_observed_without_rebinding(setup):
    s = setup
    original_context = Planner.context
    with s.audit:
        s.env.unwrapped._reset_idx([0])
        for _ in range(16):
            action = s.actor(observation(1))
            s.env.unwrapped.record_mpc_result({"solved": True})
            s.env.step(action)
        assert len(s.algorithm.update()) == 2
        with s.audit.phase("reload"):
            expected = s.actor(observation(1))
            torch.testing.assert_close(s.actor(obs=observation(1)), expected)
    report = json.loads(s.output.read_text())
    assert report["passed"] and not report["violations"]
    assert report["actor_batch_sizes"] == dict(rollout=[1] * 16, update=[8, 8], reload=[1, 1])
    assert report["counts"]["rollout"]["record_mpc_result"] == 16
    assert report["counts"]["rollout"]["env_step"] == 16
    assert report["counts"]["rollout"]["reset"] == 1
    assert report["counts"]["update"]["ppo_update"] == 1
    for phase in ("update", "reload"):
        assert all(report["counts"][phase][name] == 0 for name in TrainingAudit.mutations)
    assert s.env.unwrapped.planner.calls == dict.fromkeys(("context", "update_cloud", "consume_recovery"), 16)
    assert Planner.context is original_context
    assert "update" not in vars(s.algorithm) and "step" not in vars(s.env)
    assert "record_mpc_result" not in vars(s.env.unwrapped)
    assert not s.actor._forward_pre_hooks


@pytest.mark.parametrize("phase", ["update", "reload"])
@pytest.mark.parametrize("name", TrainingAudit.mutations)
def test_replay_rejects_each_mutation_before_original_and_keeps_failure_report(setup, phase, name):
    from isaaclab_ipc.pathcover_native import NativePathCoverFrontend

    s = setup
    world = s.env.unwrapped
    if name == "native_frontend_plan":
        world.navigation = object()
    calls = {
        "record_mpc_result": lambda: world.record_mpc_result({}),
        "env_step": lambda: s.env.step(None),
        "reset": lambda: world._reset_idx([0]),
        "planner_context": lambda: world.planner.context(3),
        "planner_update_cloud": lambda: world.planner.update_cloud([], [0, 0, 0]),
        "planner_consume_recovery": lambda: world.planner.consume_recovery(0, False),
        # The phase guard rejects before the original method can use self.
        "native_frontend_plan": lambda: NativePathCoverFrontend.plan(None, None, None, None),
    }
    original_context = Planner.context
    original_native_plan = NativePathCoverFrontend.plan
    with pytest.raises(RuntimeError, match=f"forbids {name} during {phase}"), s.audit, s.audit.phase(phase):
        calls[name]()
    assert world.records == [] and world.resets == 0 and s.env.steps == 0
    assert all(value == 0 for value in world.planner.calls.values())
    report = json.loads(s.output.read_text())
    assert not report["passed"] and len(report["violations"]) == 1
    assert report["counts"][phase][name] == 1
    assert Planner.context is original_context
    assert NativePathCoverFrontend.plan is original_native_plan
    assert not s.actor._forward_pre_hooks


def test_planning_thread_pool_counts_every_original_call(setup):
    s = setup
    original_context = Planner.context
    with s.audit, ThreadPoolExecutor(max_workers=8) as pool:
        result = list(pool.map(s.env.unwrapped.planner.context, range(200)))
    assert result == [value + 17 for value in range(200)]
    assert s.audit.result["counts"]["rollout"]["planner_context"] == 200
    assert s.env.unwrapped.planner.calls["context"] == 200
    assert Planner.context is original_context


def test_optimizer_failure_restores_existing_instance_override_and_records_error(setup):
    s = setup

    def failed_update():
        raise ValueError("optimizer failed")

    s.algorithm.update = failed_update
    with pytest.raises(ValueError, match="optimizer failed"), s.audit:
        s.algorithm.update()
    assert s.algorithm.update is failed_update
    assert s.audit._phase == "rollout"
    report = json.loads(s.output.read_text())
    assert not report["passed"] and report["error"] == "ValueError: optimizer failed"
    assert report["counts"]["update"]["ppo_update"] == 1
    assert not s.actor._forward_pre_hooks


def test_caller_cannot_silence_a_forbidden_planner_update(setup):
    s = setup
    with pytest.raises(RuntimeError, match="caller caught"), s.audit, s.audit.phase("reload"):
        with pytest.raises(RuntimeError, match="forbids planner_context"):
            s.env.unwrapped.planner.context(0)
    assert not json.loads(s.output.read_text())["passed"]
    assert s.env.unwrapped.planner.calls["context"] == 0


def test_existing_evidence_is_not_overwritten_or_instrumented(setup):
    s = setup
    s.output.write_text("previous failure")
    original_context = Planner.context
    with pytest.raises(FileExistsError), s.audit:
        pytest.fail("The existing report must reject entry")
    assert s.output.read_text() == "previous failure"
    assert Planner.context is original_context and not s.actor._forward_pre_hooks


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_finite_update_check_does_not_hide_nonfinite_parameter_after_valid_change(bad):
    initial = {"weight": torch.zeros(2), "bias": torch.zeros(1)}
    current = {"weight": torch.ones(2), "bias": torch.tensor([bad])}
    with pytest.raises(RuntimeError, match="finite, nonzero"):
        _cli.cost_network_parameter_change(current, initial)


def test_parameter_change_requires_a_real_finite_update():
    initial = {"weight": torch.zeros(2)}
    with pytest.raises(RuntimeError, match="finite, nonzero"):
        _cli.cost_network_parameter_change(initial, initial)
    assert _cli.cost_network_parameter_change({"weight": torch.tensor([0.25, -0.5])}, initial) == 0.5


def test_installed_wrapper_get_observations_reads_cache_without_advancing_planner(setup):
    s = setup
    s.env.unwrapped.obs_buf = observation(1)
    wrapper = object.__new__(_cli.RslRlVecEnvWrapper)
    wrapper.env = s.env
    wrapper.num_envs = 1

    def recompute():
        s.env.unwrapped.planner.context(0)
        return observation(1)

    s.env.unwrapped._get_observations = recompute
    with s.audit, s.audit.phase("reload"):
        obs = wrapper.get_observations()
        assert obs["policy"].data_ptr() == s.env.unwrapped.obs_buf["policy"].data_ptr()
    assert s.audit.result["passed"]
    assert s.audit.result["counts"]["reload"]["planner_context"] == 0


def test_checkpoint_helper_acquires_once_before_frozen_replay(setup):
    """A refreshing getter is a boundary probe, unlike today's cached wrapper."""
    s = setup
    source = _cli.TensorDict(observation(1), batch_size=[1])
    events = []

    def get_observations():
        events.append("get")
        s.env.unwrapped.planner.context(0)
        return source

    def load(path, map_location):
        events.append(("load", path, map_location))
        # Frozen replay must not see later mutations of the environment buffer.
        source["policy"].fill_(9)

    s.env.get_observations = get_observations
    checkpoint = s.output.with_name("checkpoint.pt")
    with s.audit:
        _cli.verify_checkpoint(SimpleNamespace(load=load), s.env, s.actor, checkpoint, "cpu", s.audit)
    assert events == ["get", ("load", str(checkpoint), "cpu")]
    assert bool((source["policy"] == 9).all())
    assert s.audit.result["counts"]["rollout"]["planner_context"] == 1
    assert s.audit.result["actor_batch_sizes"] == dict(rollout=[], update=[], reload=[1, 1])
    assert all(s.audit.result["counts"]["reload"][name] == 0 for name in TrainingAudit.mutations)


@dataclass
class RawResult:
    status: torch.Tensor
    residuals: torch.Tensor
    iterations: torch.Tensor


@dataclass
class NativeResult:
    status: torch.Tensor
    errors: torch.Tensor
    iterations: torch.Tensor
    normal_cone_error: torch.Tensor
    admm: RawResult


def mixed_observation():
    obs = observation(4)
    obs["policy"][:, 0] = torch.arange(1, 5)
    obs.update(
        sfc_valid=torch.tensor([True, True, True, False]),
        outer_status=torch.tensor([0, 1, 0, -1]),
        raw_status=torch.tensor([0, 1, 0, -1]),
        braking=torch.tensor([False, True, True, True]),
    )
    return obs


class DiagnosticActor(torch.nn.Module):
    """Independent four-row policy: executing, failed, braking, invalid SFC."""

    def __init__(self):
        super().__init__()
        self.mlp = torch.nn.Linear(1, 7, bias=False)
        with torch.no_grad():
            self.mlp.weight.copy_(torch.arange(1, 8)[:, None] / 10)

    def forward(self, obs):
        costs = self.mlp(obs["policy"][:, :1])
        valid = obs["sfc_valid"]
        solved = valid & (obs["outer_status"] == 0) & (obs["raw_status"] == 0)
        self.last_result = dict(
            solved=solved,
            braking=obs["braking"],
            sfc_valid=valid,
            primal=torch.where(valid, torch.zeros(4), torch.inf),
            dual=torch.zeros(4),
            raw_qp_status=obs["raw_status"],
            solution=costs,
        )
        rows = int(valid.sum())
        self.last_native_result = NativeResult(
            obs["outer_status"][valid],
            torch.zeros(rows, 3),
            torch.ones(rows, dtype=torch.int32),
            torch.zeros(rows),
            RawResult(obs["raw_status"][valid], torch.zeros(rows, 4), torch.full((rows,), 25)),
        )
        return torch.where((solved & ~obs["braking"])[:, None], costs[:, :4], costs[:, :4] * 0)


class OptimizerAlgorithm:
    def __init__(self, actor, *, failure=False, nonfinite=False):
        self.actor = actor
        self.optimizer = torch.optim.SGD(actor.mlp.parameters(), lr=0.1, momentum=0.9)
        self.failure, self.nonfinite = failure, nonfinite

    def update(self):
        outputs = []
        for _ in range(2):
            output = self.actor(mixed_observation())
            outputs.append(output.detach().clone())
            self.optimizer.zero_grad()
            loss = output.sum() * (float("nan") if self.nonfinite else 1)
            loss.backward()
            if self.failure:
                raise RuntimeError("failed after real backward")
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.1)
            self.optimizer.step()
        return outputs


def test_phase_result_snapshots_and_real_cost_gradient_rows_are_independent(tmp_path):
    actor, env = DiagnosticActor(), Environment()
    algorithm = OptimizerAlgorithm(actor)
    output = tmp_path / "audit.json"
    audit = TrainingAudit(algorithm, env, actor, Planner, output)
    obs = mixed_observation()
    with audit:
        with torch.no_grad():
            actor(obs)
        algorithm.update()
        with audit.phase("reload"), torch.no_grad():
            actor(obs)
        obs["policy"].fill_(99)
        actor.last_result["sfc_valid"].fill_(False)
    report = json.loads(output.read_text())
    assert report["schema_version"] == 2 and report["passed"] and report["route_passed"]
    assert report["actor_summary"]["rollout"]["sfc_valid_rows"] == 3
    assert report["actor_summary"]["rollout"]["solved_rows"] == 2
    assert report["actor_summary"]["update"]["valid_unsolved_rows"] == 2
    assert report["actor_summary"]["update"]["executed_native_solved_rows"] == 2
    assert report["actor_summary"]["update"]["native_status_counts"] == {"0": 4, "1": 2}
    assert report["actor_results"]["rollout"][0]["result"]["primal"][-1] == "inf"
    for record in report["actor_results"]["update"]:
        native = record["native"]
        assert native["row_indices"] == [0, 1, 2] and native["raw_status"] == [0, 1, 0]
        (gradient,) = record["cost_gradients"]
        assert gradient["finite"] and gradient["groups"]["executed_native_solved"]["nonzero_rows"] == 1
        assert gradient["groups"]["valid_unsolved"]["nonzero_rows"] == 0
        assert gradient["groups"]["valid_solved_braking"]["nonzero_rows"] == 0
        assert gradient["groups"]["invalid_sfc"]["nonzero_rows"] == 0
        assert gradient["values"][0] == [1, 1, 1, 1, 0, 0, 0]
    evidence = report["gradient_evidence"]
    assert evidence["native_learning_gradient_observed"] and evidence["cost_backward_calls"] == 2
    for phase, artifact in report["first_unsolved"].items():
        path = Path(artifact["path"])
        assert artifact["complete"] and hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]
        saved = torch.load(path, weights_only=False)
        assert saved["phase"] == phase and saved["unsolved_rows"].tolist() == [1, 3]
        assert saved["observation"]["policy"][:, 0].tolist() == [1, 2, 3, 4]
        assert saved["result"]["sfc_valid"].tolist() == [True, True, True, False]
        assert not saved["costs"].requires_grad and saved["costs"].device.type == "cpu"
        assert saved["native_result"]["admm"]["status"].tolist() == [0, 1, 0]
    assert not actor._forward_hooks and not actor.mlp._forward_hooks
    assert all(not parameter._backward_hooks for parameter in actor.mlp.parameters())


def test_observing_gradients_does_not_change_backward_clipping_or_optimizer(tmp_path):
    reference = DiagnosticActor()
    observed = deepcopy(reference)
    plain, instrumented = OptimizerAlgorithm(reference), OptimizerAlgorithm(observed)
    expected = plain.update()
    audit = TrainingAudit(instrumented, Environment(), observed, Planner, tmp_path / "audit.json")
    with audit:
        actual = instrumented.update()
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    for a, b in zip(observed.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
        torch.testing.assert_close(a.grad, b.grad, atol=0, rtol=0)
    assert all(event["max_abs"] == 1 for event in audit.result["parameter_gradients"]["update"])
    assert float(next(observed.parameters()).grad.abs().max()) < 0.1
    assert all(audit.result["counts"]["update"][name] == 0 for name in TrainingAudit.mutations)


def test_failure_after_forward_backward_keeps_replay_artifact_and_restores_hooks(tmp_path):
    actor = DiagnosticActor()
    algorithm = OptimizerAlgorithm(actor, failure=True)
    output = tmp_path / "audit.json"
    with (
        pytest.raises(RuntimeError, match="real backward"),
        TrainingAudit(algorithm, Environment(), actor, Planner, output),
    ):
        algorithm.update()
    report = json.loads(output.read_text())
    assert not report["passed"] and report["first_unsolved"]["update"]["complete"]
    assert report["gradient_evidence"]["cost_backward_calls"] == 1
    assert report["error"] == "RuntimeError: failed after real backward"
    assert not actor._forward_hooks and all(not parameter._backward_hooks for parameter in actor.mlp.parameters())


def test_nonfinite_gradients_are_evidence_failure_without_changing_route_gate(tmp_path):
    actor = DiagnosticActor()
    algorithm = OptimizerAlgorithm(actor, nonfinite=True)
    output = tmp_path / "audit.json"
    with TrainingAudit(algorithm, Environment(), actor, Planner, output):
        algorithm.update()
    report = json.loads(output.read_text())
    assert report["passed"] and report["route_passed"]
    assert not report["gradient_evidence"]["all_observed_gradients_finite"]
    assert not report["gradient_evidence"]["native_learning_gradient_observed"]
    assert report["actor_results"]["update"][0]["cost_gradients"][0]["values"][0][0] == "nan"
    assert torch.isnan(next(actor.parameters()).grad).any()


def test_momentum_parameter_change_is_not_mistaken_for_current_native_gradient(tmp_path):
    actor = DiagnosticActor()
    algorithm = OptimizerAlgorithm(actor)
    obs = mixed_observation()
    obs["outer_status"].fill_(1)
    obs["raw_status"].fill_(1)
    obs["braking"].fill_(True)
    initial = {name: value.clone() for name, value in actor.mlp.state_dict().items()}
    for parameter in actor.mlp.parameters():
        algorithm.optimizer.state[parameter]["momentum_buffer"] = torch.ones_like(parameter)

    def update():
        algorithm.optimizer.zero_grad()
        actor(obs).sum().backward()
        algorithm.optimizer.step()

    algorithm.update = update
    output = tmp_path / "audit.json"
    with TrainingAudit(algorithm, Environment(), actor, Planner, output):
        algorithm.update()
    report = json.loads(output.read_text())
    assert _cli.cost_network_parameter_change(actor.mlp.state_dict(), initial) > 0
    assert report["passed"] and report["actor_summary"]["update"]["valid_unsolved_rows"] == 3
    evidence = report["gradient_evidence"]
    assert evidence["all_observed_gradients_finite"] and evidence["cost_backward_calls"] == 1
    assert evidence["parameter_nonzero_events"] == 0
    assert not evidence["native_learning_gradient_observed"]


def test_existing_replay_artifact_rejects_entry_without_overwrite_or_hooks(tmp_path):
    output = tmp_path / "audit.json"
    artifact = tmp_path / "audit.first-unsolved-update.pt"
    artifact.write_bytes(b"prior evidence")
    actor = DiagnosticActor()
    with (
        pytest.raises(FileExistsError),
        TrainingAudit(OptimizerAlgorithm(actor), Environment(), actor, Planner, output),
    ):
        pytest.fail("Existing replay evidence must be preserved")
    assert artifact.read_bytes() == b"prior evidence" and not output.exists()
    assert not actor._forward_pre_hooks and not actor._forward_hooks


def test_installed_rsl_logger_keeps_actual_reward_history_for_each_step():
    from isaaclab_ipc.tasks.mpc_env import IPCMPCEnv

    positions = torch.tensor([[0.1, 0.0, 0.0]])
    previous_distance = torch.ones(1)
    world = SimpleNamespace(
        body_state=lambda: (positions, None),
        scene=SimpleNamespace(env_origins=torch.zeros(1, 3)),
        goal=torch.tensor([[1.0, 0.0, 0.0]]),
        hit=torch.tensor([False]),
        reward_active=torch.tensor([True]),
        extras={"success": torch.tensor([False]), "log": {"Episode/test": torch.tensor(0.25)}},
        command_violations=_cli.np.zeros(1, dtype=int),
        device="cpu",
        reset_terminated=torch.tensor([False]),
        reset_time_outs=torch.tensor([False]),
        cfg=SimpleNamespace(reward=NavigationRewardCfg(), log_dir=None),
        previous_distance=previous_distance,
        controller=SimpleNamespace(command=torch.zeros(1, 4)),
        previous_action=torch.zeros(1, 4),
        step_dt=0.01,
        reward_sums=torch.zeros(1, 4),
        recorder_manager=SimpleNamespace(record_post_step=lambda: None),
    )
    logger = Logger(None, {"algorithm": {}, "num_steps_per_env": 2}, {}, 1, False, 1, 0, "cpu")
    logged = {}
    logger.writer = SimpleNamespace(add_scalar=lambda key, value, _step: logged.update({key: float(value)}))
    for index, solved in enumerate((True, False)):
        positions[0, 0] = 0.1 * (index + 1)
        world.last_mpc_result = dict(
            solved=torch.tensor([solved]), braking=torch.tensor([not solved]), sfc_valid=torch.tensor([True])
        )
        reward = IPCMPCEnv._get_rewards(world)
        torch.testing.assert_close(reward, torch.tensor([0.0099]))
        logger.process_env_step(reward, torch.zeros(1, dtype=torch.bool), world.extras)
    assert world.previous_distance.data_ptr() == previous_distance.data_ptr()
    assert logger.ep_extras[0] is not logger.ep_extras[1]
    assert [float(item["Control/solved"]) for item in logger.ep_extras] == [1, 0]
    logger.log(0, 0, 1, 1, 1, {}, 1e-4, torch.tensor([0.03]), None)
    assert logged["Control/solved"] == 0.5 and logged["Control/braking"] == 0.5
    assert logged["Episode/test"] == 0.25
