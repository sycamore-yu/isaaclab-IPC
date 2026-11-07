"""Train or evaluate CUDA AC-MPC on the standard IPC Isaac Lab tasks."""

import argparse
import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import fields, is_dataclass
from functools import wraps
from importlib.metadata import version
from pathlib import Path
from threading import RLock
from time import perf_counter

import gymnasium as gym
import numpy as np
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils.hydra import resolve_task_config
from tensordict import TensorDict

import isaaclab_ipc.tasks  # noqa: F401


def _audit_snapshot(value):
    """Copy observed data without retaining graphs or calling the policy again."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _audit_snapshot(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, (Mapping, TensorDict)):
        return {key: _audit_snapshot(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_audit_snapshot(item) for item in value]
    return value


def _audit_json(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {key: _audit_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_audit_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def _gradient_stats(gradient):
    value = gradient.detach().cpu().double()
    finite = bool(torch.isfinite(value).all())
    return dict(
        shape=list(value.shape),
        finite=finite,
        nonzero=finite and bool((value != 0).any()),
        max_abs=float(value.abs().max()) if value.numel() else 0.0,
        l2=float(value.norm()),
    )


class TrainingAudit:
    """Observe one training run without changing PPO or Planner behavior.

    The runner finishes its planning worker calls before entering PPO.update.
    One locked phase therefore also covers the existing planning thread pool.
    Class hooks cover planners replaced by automatic environment resets.
    """

    phases = ("rollout", "update", "reload")
    mutations = (
        "record_mpc_result",
        "env_step",
        "reset",
        "planner_context",
        "planner_update_cloud",
        "planner_consume_recovery",
        "native_frontend_plan",
    )

    def __init__(self, algorithm, env, actor, planner_type, output):
        self.algorithm, self.env, self.actor = algorithm, env, actor
        self.planner_type, self.output = planner_type, Path(output)
        self._phase = "rollout"
        self._lock = RLock()
        self._stack = ExitStack()
        names = (*self.mutations, "actor_forward", "ppo_update")
        self.counts = {phase: dict.fromkeys(names, 0) for phase in self.phases}
        self.actor_batch_sizes = {phase: [] for phase in self.phases}
        self.actor_results = {phase: [] for phase in self.phases}
        self.parameter_gradients = {phase: [] for phase in self.phases}
        self.first_unsolved = {}
        self._actor_calls = []
        self.violations = []
        self.result = None

    def _record(self, name, batch_size=None):
        with self._lock:
            phase = self._phase
            self.counts[phase][name] += 1
            if batch_size is not None:
                self.actor_batch_sizes[phase].append(batch_size)
            if phase != "rollout" and name in self.mutations:
                message = f"Training audit forbids {name} during {phase}"
                self.violations.append(message)
                # Reject before calling the original method, including worker calls.
                raise RuntimeError(message)

    @contextmanager
    def phase(self, name):
        if name not in self.phases:
            raise ValueError(f"Unknown training audit phase: {name}")
        with self._lock:
            previous, self._phase = self._phase, name
        try:
            yield
        finally:
            with self._lock:
                self._phase = previous

    def _patch(self, owner, name, factory):
        # Instance originals are already bound; class originals are not. A
        # plain forwarding function preserves both conventions without rebinding.
        existed = name in vars(owner)
        previous = vars(owner).get(name)
        original = getattr(owner, name)
        setattr(owner, name, factory(original))
        self._stack.callback(setattr if existed else delattr, owner, name, *([previous] if existed else []))

    def _guard(self, name):
        def factory(original):
            @wraps(original)
            def call(*args, **kwargs):
                self._record(name)
                return original(*args, **kwargs)

            return call

        return factory

    def _update(self, original):
        @wraps(original)
        def call(*args, **kwargs):
            with self.phase("update"):
                self._record("ppo_update")
                return original(*args, **kwargs)

        return call

    def _actor_forward(self, _module, args, kwargs):
        obs = args[0] if args else kwargs["obs"]
        self._record("actor_forward", int(obs["policy"].shape[0]))
        with self._lock:
            record = dict(
                phase=self._phase,
                forward_index=self.counts[self._phase]["actor_forward"],
                batch_size=int(obs["policy"].shape[0]),
                completed=False,
                cost_gradients=[],
            )
            self.actor_results[self._phase].append(record)
            self._actor_calls.append(dict(record=record, costs=None))

    @staticmethod
    def _row_groups(record):
        result = record.get("result") or {}
        if not all(name in result for name in ("solved", "braking", "sfc_valid")):
            return {"unknown": torch.ones(record["batch_size"], dtype=torch.bool)}
        solved, braking, valid = (
            torch.tensor(result[name], dtype=torch.bool) for name in ("solved", "braking", "sfc_valid")
        )
        native_solved = torch.zeros_like(valid)
        native = record.get("native")
        if native is not None:
            indexes = torch.tensor(native["row_indices"], dtype=torch.long)
            native_solved[indexes] = (torch.tensor(native["status"]) == 0) & (torch.tensor(native["raw_status"]) == 0)
        return dict(
            valid_solved_unbraked=valid & solved & ~braking,
            valid_solved_braking=valid & solved & braking,
            valid_unsolved=valid & ~solved,
            invalid_sfc=~valid,
            executed_native_solved=valid & solved & ~braking & native_solved,
        )

    def _cost_forward(self, _module, _args, costs):
        if not self._actor_calls:
            return
        frame = self._actor_calls[-1]
        frame["costs"] = costs
        record = frame["record"]
        record["cost_requires_grad"] = costs.requires_grad
        if costs.requires_grad:

            def observe(gradient):
                value = gradient.detach().cpu()
                finite_rows = torch.isfinite(value).flatten(1).all(-1)
                nonzero_rows = finite_rows & (value != 0).flatten(1).any(-1)
                with self._lock:
                    groups = {
                        name: dict(
                            rows=int(mask.sum()),
                            nonzero_rows=int((mask & nonzero_rows).sum()),
                            nonfinite_rows=int((mask & ~finite_rows).sum()),
                        )
                        for name, mask in self._row_groups(record).items()
                    }
                    record["cost_gradients"].append(
                        dict(
                            backward_phase=self._phase,
                            update_call=self.counts["update"]["ppo_update"],
                            **_gradient_stats(value),
                            values=_audit_json(value),
                            groups=groups,
                        )
                    )
                # Returning None leaves the incoming gradient untouched.

            self._stack.callback(costs.register_hook(observe).remove)

    def _parameter_gradient(self, name):
        def observe(gradient):
            with self._lock:
                self.parameter_gradients[self._phase].append(
                    dict(name=name, update_call=self.counts["update"]["ppo_update"], **_gradient_stats(gradient))
                )

        return observe

    def _actor_result(self, module, args, kwargs, _output):
        frame = self._actor_calls.pop()
        record, costs = frame["record"], frame["costs"]
        record["completed"] = True
        result = getattr(module, "last_result", None)
        if result is None:
            record["result"] = None
            return
        names = ("solved", "braking", "sfc_valid", "primal", "dual", "normal_cone", "raw_qp_status", "iterations")
        record["result"] = _audit_json({name: result[name] for name in names if name in result})
        native = getattr(module, "last_native_result", None)
        if native is not None:
            rows = result["sfc_valid"].flatten().nonzero().flatten()
            if len(rows) != len(native.status):
                raise RuntimeError("Training audit cannot map native results to observation rows")
            record["native"] = _audit_json(
                dict(
                    row_indices=rows,
                    status=native.status,
                    errors=native.errors,
                    sqp_iterations=native.iterations,
                    raw_status=native.admm.status,
                    raw_residuals=native.admm.residuals,
                    admm_iterations=native.admm.iterations,
                    normal_cone_error=native.normal_cone_error,
                )
            )
        phase = record["phase"]
        unsolved = ~result["solved"].flatten().bool()
        if bool(unsolved.any()) and phase not in self.first_unsolved:
            path = self.output.with_name(f"{self.output.stem}.first-unsolved-{phase}.pt")
            evidence = dict(path=str(path.resolve()), complete=False)
            self.first_unsolved[phase] = evidence
            obs = args[0] if args else kwargs["obs"]
            snapshot = _audit_snapshot(
                dict(
                    schema_version=2,
                    phase=phase,
                    forward_index=record["forward_index"],
                    unsolved_rows=unsolved.nonzero().flatten(),
                    native_row_indices=result["sfc_valid"].flatten().nonzero().flatten(),
                    observation=obs,
                    costs=costs,
                    result=result,
                    native_result=native,
                    solver_config=module.solver_metadata(),
                )
            )
            with path.open("xb") as handle:
                torch.save(snapshot, handle)
            with path.open("rb") as handle:
                evidence.update(complete=True, sha256=hashlib.file_digest(handle, "sha256").hexdigest())

    def _result_summary(self, records):
        result = dict(forward_calls=len(records), rows=sum(item["batch_size"] for item in records))
        result["completed_calls"] = sum(item["completed"] for item in records)
        for name in ("solved", "braking", "sfc_valid"):
            result[name + "_rows"] = sum(sum((item.get("result") or {}).get(name, [])) for item in records)
        result["native_rows"] = sum(len(item.get("native", {}).get("row_indices", [])) for item in records)
        result["missing_result_rows"] = sum(item["batch_size"] for item in records if not item.get("result"))
        for name in (
            "valid_solved_unbraked",
            "valid_solved_braking",
            "valid_unsolved",
            "invalid_sfc",
            "executed_native_solved",
        ):
            result[name + "_rows"] = sum(
                int(self._row_groups(item).get(name, torch.zeros(0)).sum()) for item in records
            )
        for name in ("status", "raw_status"):
            result["native_" + name + "_counts"] = dict(
                Counter(value for item in records for value in item.get("native", {}).get(name, []))
            )
        return result

    def _gradient_evidence(self):
        costs = [
            gradient
            for record in self.actor_results["update"]
            for gradient in record["cost_gradients"]
            if gradient["backward_phase"] == "update"
        ]
        parameters = self.parameter_gradients["update"]
        native_rows = sum(item["groups"].get("executed_native_solved", {}).get("nonzero_rows", 0) for item in costs)
        finite = bool(costs and parameters) and all(item["finite"] for item in [*costs, *parameters])
        return dict(
            scope="Observed PPO loss gradients before clipping/optimizer; repeated row evaluations are counted",
            cost_backward_calls=len(costs),
            parameter_gradient_events=len(parameters),
            all_observed_gradients_finite=finite,
            parameter_nonzero_events=sum(item["nonzero"] for item in parameters),
            executed_native_rows_with_nonzero_cost_gradient=native_rows,
            native_learning_gradient_observed=finite
            and native_rows > 0
            and any(item["nonzero"] for item in parameters),
        )

    def __enter__(self):
        for path in (
            self.output,
            *(self.output.with_name(f"{self.output.stem}.first-unsolved-{phase}.pt") for phase in self.phases),
        ):
            if path.exists() or path.is_symlink():
                raise FileExistsError(path)
        try:
            self._patch(self.env.unwrapped, "record_mpc_result", self._guard("record_mpc_result"))
            self._patch(self.env, "step", self._guard("env_step"))
            self._patch(self.env.unwrapped, "_reset_idx", self._guard("reset"))
            if self.planner_type is not None:
                for name in ("context", "update_cloud", "consume_recovery"):
                    self._patch(self.planner_type, name, self._guard("planner_" + name))
            if getattr(self.env.unwrapped, "navigation", None) is not None:
                from isaaclab_ipc.pathcover_native import NativePathCoverFrontend

                self._patch(NativePathCoverFrontend, "plan", self._guard("native_frontend_plan"))
            self._patch(self.algorithm, "update", self._update)
            hook = self.actor.register_forward_pre_hook(self._actor_forward, with_kwargs=True)
            self._stack.callback(hook.remove)
            self._stack.callback(self.actor.register_forward_hook(self._actor_result, with_kwargs=True).remove)
            if hasattr(self.actor, "mlp"):
                self._stack.callback(self.actor.mlp.register_forward_hook(self._cost_forward).remove)
                for name, parameter in self.actor.mlp.named_parameters():
                    if parameter.requires_grad:
                        self._stack.callback(parameter.register_hook(self._parameter_gradient(name)).remove)
        except BaseException as error:
            self.__exit__(type(error), error, error.__traceback__)
            raise
        return self

    def _write(self, error):
        with self._lock:
            self.result = dict(
                schema_version=2,
                scope="Route gate only; solver/gradient evidence is separate and does not establish navigation acceptance",
                passed=error is None and not self.violations,
                route_passed=error is None and not self.violations,
                row_count_semantics="Forward evaluations, including replay repeats; first-unsolved artifacts include invalid SFC rows",
                counts=self.counts,
                actor_batch_sizes=self.actor_batch_sizes,
                actor_results=self.actor_results,
                actor_summary={phase: self._result_summary(records) for phase, records in self.actor_results.items()},
                parameter_gradients=self.parameter_gradients,
                gradient_evidence=self._gradient_evidence(),
                first_unsolved=self.first_unsolved,
                violations=self.violations,
                error=None if error is None else f"{type(error).__name__}: {error}",
            )
            self.result = _audit_json(self.result)
            with self.output.open("x") as stream:
                json.dump(self.result, stream, indent=2, allow_nan=False)

    def __exit__(self, _kind, error, _traceback):
        try:
            self._stack.close()
        except BaseException as restore_error:
            self._write(restore_error)
            raise
        finally:
            self._actor_calls.clear()
        if error is None and self.violations:
            error = RuntimeError("Training audit recorded a forbidden call that its caller caught")
            self._write(error)
            raise error
        self._write(error)
        return False


def cost_network_parameter_change(current, initial):
    changes = [(current[key] - value).abs().max().item() for key, value in initial.items()]
    if not all(math.isfinite(change) for change in changes) or max(changes) <= 0:
        raise RuntimeError("Training must make a finite, nonzero change to the cost network")
    return max(changes)


def verify_checkpoint(runner, env, actor, checkpoint, device, audit=None):
    """Compare checkpoint actions using one cloned observation."""
    # Today's RSL wrapper reads obs_buf without advancing Planner. Keep the
    # single acquisition outside reload so that only frozen replay is audited.
    snapshot = env.get_observations().clone()
    with audit.phase("reload") if audit is not None else nullcontext():
        with torch.no_grad():
            expected = actor(snapshot).clone()
        runner.load(str(checkpoint), map_location=device)
        with torch.no_grad():
            restored = actor(snapshot)
        torch.testing.assert_close(restored, expected, atol=1e-6, rtol=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "evaluate"))
    parser.add_argument("--task", default="Isaac-ACMPC-CTBR-Direct")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument(
        "--episodes", type=int, default=1, help="Evaluation episodes per environment; consecutive seeds"
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--audit-training", action="store_true", help="Assert rollout-only Planner mutation and record phase counts"
    )
    parser.add_argument("--record", type=Path, help="Write one forest episode with SFC to a new RRD file")
    parser.add_argument(
        "--fixed-cost", action="store_true", help="Evaluate this task's fixed MPC cost without a checkpoint"
    )
    args, overrides = parser.parse_known_args()
    if args.audit_training and args.mode != "train":
        parser.error("--audit-training is only available in train mode")
    if args.episodes < 1 or (args.mode == "train" and args.episodes != 1):
        parser.error("--episodes must be positive and is only used for evaluation")
    if args.record and args.episodes != 1:
        parser.error("--record requires one episode")
    if args.fixed_cost and (args.mode != "evaluate" or args.checkpoint):
        parser.error("--fixed-cost selects evaluation without a checkpoint")
    if args.mode == "evaluate" and not (args.fixed_cost or args.checkpoint):
        parser.error("Evaluation needs --checkpoint or --fixed-cost")
    if args.record and (args.mode != "evaluate" or args.record.exists()):
        parser.error("--record requires evaluation and a new RRD path")
    args.output.mkdir(parents=True, exist_ok=False)
    env_cfg, agent_cfg = resolve_task_config(args.task, "rsl_rl_cfg_entry_point", overrides=overrides)
    if args.num_envs is not None:
        env_cfg.scene.num_envs = args.num_envs
    if args.record and (env_cfg.scene.num_envs != 1 or env_cfg.scenario != "forest"):
        parser.error("--record requires one forest environment")
    runner_config = handle_deprecated_rsl_rl_cfg(agent_cfg, version("rsl-rl-lib"))
    if getattr(env_cfg, "ctbr_model", None) is not None:
        from isaaclab_ipc.rl.ctbr_observation import ctbr_feature_size

        runner_config.actor.model_config = dict(env_cfg.ctbr_model)
        runner_config.actor.cost_nodes = env_cfg.ctbr_model["cost_nodes"]
        runner_config.actor.prediction_dt = env_cfg.ctbr_model["dt"]
        env_cfg.observation_space = ctbr_feature_size(env_cfg.ctbr_model["cost_nodes"])
    agent_cfg = runner_config.to_dict()
    env_cfg.seed = agent_cfg["seed"]
    device = env_cfg.sim.device
    if not str(device).startswith("cuda"):
        parser.error("This training entry point requires the CUDA solver and Newton backend")
    if args.iterations is not None:
        agent_cfg["max_iterations"] = args.iterations
    torch.manual_seed(agent_cfg["seed"])
    np.random.seed(agent_cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    agent_cfg["device"] = device
    env_cfg.log_dir = str(args.output.resolve())
    env_cfg.automatic_reset = args.mode == "train"
    env_cfg.rerun_path = str(args.record) if args.record else None
    env_cfg.rerun_label = "fixed MPC cost" if args.fixed_cost else "learned cost"
    config = dict(task=args.task, runner=agent_cfg, environment=env_cfg.to_dict())
    (args.output / "config.json").write_text(json.dumps(config, indent=2, default=str))
    env = None
    actor = None
    audit = None
    failure = None
    try:
        env = RslRlVecEnvWrapper(gym.make(args.task, cfg=env_cfg))
        runner_config.device = device
        runner_config.max_iterations = agent_cfg["max_iterations"]
        runner = create_rsl_rl_runner(env, runner_config, str(args.output) if args.mode == "train" else None)
        runner.add_git_repo_to_log(__file__)
        if args.checkpoint:
            infos = runner.load(str(args.checkpoint), map_location=device)
            if infos and "torch_rng" in infos and args.mode == "train":
                torch.set_rng_state(infos["torch_rng"].cpu())
                torch.cuda.set_rng_state(infos["cuda_rng"].cpu(), device)
                np.random.set_state(infos["numpy_rng"])
            runner.current_learning_iteration += 1
        actor = runner.alg.get_policy()
        actor.fixed_cost = args.fixed_cost
        initial = {k: v.detach().clone() for k, v in actor.mlp.state_dict().items()}
        start = perf_counter()
        if args.mode == "train":
            if args.audit_training:
                planner_type = None
                if hasattr(env.unwrapped, "planners"):
                    from isaaclab_ipc.planner import Planner

                    planner_type = Planner
                audit = TrainingAudit(runner.alg, env, actor, planner_type, args.output / "training-audit.json")
            with audit if audit is not None else nullcontext():
                runner.learn(agent_cfg["max_iterations"])
                runner.save(
                    str(args.output / "checkpoint.pt"),
                    infos=dict(
                        torch_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state(device),
                        numpy_rng=np.random.get_state(),
                        task=args.task,
                        runner_cfg=agent_cfg,
                        resume_semantics="model, optimizer and RNG; fresh environment episodes",
                    ),
                )
                verify_checkpoint(runner, env, actor, args.output / "checkpoint.pt", device, audit)
                delta = cost_network_parameter_change(actor.mlp.state_dict(), initial)
                metrics = dict(
                    cost_network_max_parameter_change=delta,
                    checkpoint_action_match=True,
                    last_iteration=runner.current_learning_iteration,
                    completed_episodes=len(env.unwrapped.records),
                )
            if audit is not None:
                metrics["training_audit"] = audit.result
        else:
            actor.eval()
            times = []
            for repeat in range(args.episodes):
                seed = agent_cfg["seed"] + repeat
                values, _ = env.unwrapped.reset(seed=seed)
                obs = TensorDict(values, batch_size=[env.num_envs])
                first_record = len(env.unwrapped.records)
                failure_saved = False
                for step_index in range(env.max_episode_length + 1):
                    with torch.inference_mode():
                        torch.cuda.synchronize()
                        tick = perf_counter()
                        try:
                            actions = actor(obs)
                        except (ValueError, RuntimeError):
                            torch.save(obs.cpu(), args.output / "failed-observation.pt")
                            raise
                        torch.cuda.synchronize()
                        solve_ms = (perf_counter() - tick) * 1000
                        unsolved = ~actor.last_result["solved"] & actor.last_result["sfc_valid"]
                        if unsolved.any() and not failure_saved:
                            costs = actor.mlp(actor.get_latent(obs))
                            problem = actor.problem_snapshot(obs[unsolved], costs[unsolved])
                            torch.save(
                                dict(
                                    observation=obs.cpu(),
                                    solver_config=actor.solver_metadata(),
                                    problem_rows=unsolved.nonzero().flatten().cpu(),
                                    problem={name: value.detach().cpu() for name, value in problem.items()},
                                    result={
                                        k: v.cpu() if isinstance(v, torch.Tensor) else v
                                        for k, v in actor.last_result.items()
                                    },
                                    costs=costs.detach().cpu(),
                                    native_row_indices=actor.last_result["sfc_valid"]
                                    .flatten()
                                    .nonzero()
                                    .flatten()
                                    .cpu(),
                                    native_result=_audit_snapshot(getattr(actor, "last_native_result", None)),
                                ),
                                args.output / f"first-unsolved-seed-{seed}.pt",
                            )
                            failure_saved = True
                        env.unwrapped.record_mpc_result(actor.last_result)
                        obs, _, _, _ = env.step(actions)
                        torch.cuda.synchronize()
                        times.append([solve_ms, (perf_counter() - tick) * 1000])
                    if (step_index + 1) % 100 == 0:
                        print(
                            json.dumps(
                                dict(
                                    evaluation_seed=seed,
                                    steps=step_index + 1,
                                    position=obs["ctbr_state" if "ctbr_state" in obs else "mpc_state"][:, :3].cpu().tolist(),
                                    solved=actor.last_result["solved"].cpu().tolist(),
                                    braking=actor.last_result["braking"].cpu().tolist(),
                                )
                            ),
                            flush=True,
                        )
                    if env.unwrapped.reported.all():
                        break
                records = env.unwrapped.records[first_record:]
                if len(records) != env.num_envs:
                    raise RuntimeError(f"Incomplete evaluation seed {seed}: {len(records)}/{env.num_envs}")
                for record in records:
                    record["evaluation_seed"] = seed
                (args.output / "records.json").write_text(json.dumps(env.unwrapped.records, indent=2))
                print(
                    json.dumps(
                        dict(evaluation_seed=seed, completed=len(records), success=sum(r["success"] for r in records))
                    ),
                    flush=True,
                )
            np.save(args.output / "timing-ms.npy", np.asarray(times))
            records = env.unwrapped.records
            metrics = dict(
                policy=("fixed-ctbr" if actor.solver_metadata()["dynamics"] == "ac-mpc-ctbr" else "fixed-ipc")
                if args.fixed_cost
                else "learned-cost",
                checkpoint=str(args.checkpoint),
                requested_episodes=args.episodes * env.num_envs,
                evaluation_seeds=list(range(agent_cfg["seed"], agent_cfg["seed"] + args.episodes)),
                completed_episodes=len(records),
                success=sum(r["success"] for r in records),
                actor_mean_ms=float(np.mean(times, axis=0)[0]),
                step_mean_ms=float(np.mean(times, axis=0)[1]),
                timing_includes_first_call_compilation=True,
            )
        metrics["mpc"] = actor.solver_metadata()
        if hasattr(env.unwrapped, "ctbr_runtime_contract"):
            metrics["physical_model_contract"] = env.unwrapped.ctbr_runtime_contract
        metrics["wall_s"] = perf_counter() - start
        metrics["frontend_mean_ms"] = float(np.mean(env.unwrapped.frontend_times)) if env.unwrapped.frontend_times else 0.0
        navigation = getattr(env.unwrapped, "navigation", None)
        if navigation is not None:
            metrics["frontend_reference_quality"] = navigation.reference_quality
            batches = navigation.frontend_batch_times_ms
            metrics["frontend_batch_time_ms"] = {
                "count": len(batches),
                "mean": float(np.mean(batches)) if batches else None,
                "p95": float(np.percentile(batches, 95)) if batches else None,
                "scope": "All environments: native plans and reference/SFC sampling",
            }
        (args.output / "records.json").write_text(json.dumps(env.unwrapped.records, indent=2))
        (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2))
        print(json.dumps(metrics), flush=True)
    except BaseException as error:
        failure = error
        raise
    finally:
        try:
            if args.audit_training and audit is None:
                with (args.output / "training-audit.json").open("x") as stream:
                    json.dump(
                        dict(passed=False, counts=None, error=f"Setup failed before audit: {failure}"), stream, indent=2
                    )
        finally:
            try:
                if actor is not None:
                    actor.close()
            finally:
                if env is not None:
                    try:
                        navigation = getattr(env.unwrapped, "navigation", None)
                        if navigation is not None:
                            (args.output / "reference-quality.json").write_text(
                                json.dumps(navigation.reference_quality, indent=2, allow_nan=False)
                            )
                    finally:
                        env.close()


if __name__ == "__main__":
    main()
