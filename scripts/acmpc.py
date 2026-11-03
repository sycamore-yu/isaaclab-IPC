"""Train or evaluate CUDA AC-MPC on the standard IPC Isaac Lab tasks."""

import argparse
import json
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import gymnasium as gym
import numpy as np
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils.hydra import resolve_task_config
from tensordict import TensorDict

import isaaclab_ipc.tasks  # noqa: F401


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "evaluate"))
    parser.add_argument("--task", default="Isaac-IPC-Avoid-ACMPC-Direct")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument(
        "--episodes", type=int, default=1, help="Evaluation episodes per environment; consecutive seeds"
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--record", type=Path, help="Write one forest episode with SFC to a new RRD file")
    parser.add_argument(
        "--fixed-cost", action="store_true", help="Evaluate the fixed IPC cost with the same CUDA solver"
    )
    args, overrides = parser.parse_known_args()
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
    env_cfg.rerun_label = "fixed IPC cost" if args.fixed_cost else "learned cost"
    config = dict(task=args.task, runner=agent_cfg, environment=env_cfg.to_dict())
    (args.output / "config.json").write_text(json.dumps(config, indent=2, default=str))
    env = RslRlVecEnvWrapper(gym.make(args.task, cfg=env_cfg))
    try:
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
        initial = {k: v.detach().clone() for k, v in actor.mlp.state_dict().items()}
        start = perf_counter()
        if args.mode == "train":
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
            # Use the same frozen observation to verify checkpoint consumption.
            snapshot = env.get_observations().clone()
            with torch.no_grad():
                expected = actor(snapshot).clone()
            runner.load(str(args.output / "checkpoint.pt"), map_location=device)
            with torch.no_grad():
                restored = actor(snapshot)
            torch.testing.assert_close(restored, expected, atol=1e-6, rtol=0)
            delta = max((actor.mlp.state_dict()[k] - v).abs().max().item() for k, v in initial.items())
            if delta == 0:
                raise RuntimeError("Training did not update the cost network")
            metrics = dict(
                cost_network_max_parameter_change=delta,
                checkpoint_action_match=True,
                last_iteration=runner.current_learning_iteration,
                completed_episodes=len(env.unwrapped.records),
            )
        else:
            actor.eval()
            times = []
            for repeat in range(args.episodes):
                seed = agent_cfg["seed"] + repeat
                values, _ = env.unwrapped.reset(seed=seed)
                obs = TensorDict(values, batch_size=[env.num_envs])
                first_record = len(env.unwrapped.records)
                failure_saved = False
                for _ in range(env.max_episode_length + 1):
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
                            problem = actor.problem(obs, costs)
                            torch.save(
                                dict(
                                    observation=obs.cpu(),
                                    problem=[v.cpu() for v in problem],
                                    result={
                                        k: v.cpu() if isinstance(v, torch.Tensor) else v
                                        for k, v in actor.last_result.items()
                                    },
                                ),
                                args.output / f"first-unsolved-seed-{seed}.pt",
                            )
                            failure_saved = True
                        env.unwrapped.record_mpc_result(actor.last_result)
                        obs, _, _, _ = env.step(actions)
                        torch.cuda.synchronize()
                        times.append([solve_ms, (perf_counter() - tick) * 1000])
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
                policy="fixed-ipc" if args.fixed_cost else "learned-cost",
                checkpoint=str(args.checkpoint),
                requested_episodes=args.episodes * env.num_envs,
                evaluation_seeds=list(range(agent_cfg["seed"], agent_cfg["seed"] + args.episodes)),
                completed_episodes=len(records),
                success=sum(r["success"] for r in records),
                actor_mean_ms=float(np.mean(times, axis=0)[0]),
                step_mean_ms=float(np.mean(times, axis=0)[1]),
            )
        metrics["wall_s"] = perf_counter() - start
        metrics["frontend_mean_ms"] = float(np.mean(env.unwrapped.frontend_times))
        (args.output / "records.json").write_text(json.dumps(env.unwrapped.records, indent=2))
        (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2))
        print(json.dumps(metrics), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
