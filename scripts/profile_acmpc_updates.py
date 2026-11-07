#!/usr/bin/env python3
"""Diagnose actual RSL-RL PPO update time without replacing its training loop.

CUDA synchronization, frozen minibatch writes, and recompilation counters make
this a diagnostic run. Use benchmark_acmpc_updates.py for throughput results.
Set TORCH_LOGS=recompiles to retain the exact Dynamo guard failures in the log.
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import torch
from acmpc import main as acmpc_main
from rsl_rl.algorithms import PPO

from isaaclab_ipc.rl.ctbr_actor import CTBRMPCActor
from isaaclab_ipc.turbompc.solver import TorchMPC, _ImplicitTrajectory


def compile_counters():
    from torch._dynamo.utils import counters

    return {name: dict(values) for name, values in counters.items() if values}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=4)
    args = parser.parse_args()
    events = []
    update_count = 0
    phase = None
    minibatch = 0
    original_actor = CTBRMPCActor.forward
    original_solve = TorchMPC.solve
    original_adjoint = _ImplicitTrajectory.backward
    original_backward = torch.Tensor.backward
    original_step = torch.optim.Adam.step
    original_update = PPO.update

    def timed(name, function, *values, **kwargs):
        torch.cuda.synchronize()
        tick = perf_counter()
        result = function(*values, **kwargs)
        torch.cuda.synchronize()
        events.append(dict(kind=name, update=phase, minibatch=minibatch, milliseconds=1000 * (perf_counter() - tick)))
        return result

    def actor_forward(actor, obs, *values, **kwargs):
        nonlocal minibatch
        if phase is None:
            return original_actor(actor, obs, *values, **kwargs)
        minibatch += 1
        captured = {}

        def costs_hook(module, inputs, output):
            captured["costs"] = output.detach()

        with actor.mlp.register_forward_hook(costs_hook):
            result = timed("actor_forward", original_actor, actor, obs, *values, **kwargs)
        native = actor.last_native_result
        sqp, admm = native.iterations.cpu(), native.admm.iterations.cpu()
        statuses, counts = native.status.cpu().unique(return_counts=True)
        raw_statuses, raw_counts = native.admm.status.cpu().unique(return_counts=True)
        events.append(dict(
            kind="solver_result", update=phase, minibatch=minibatch, batch_size=len(obs),
            sqp_iterations_median=float(sqp.double().median()), sqp_iterations_max=int(sqp.max()),
            admm_iterations_median=float(admm.double().median()), admm_iterations_max=int(admm.max()),
            status_counts=dict(zip(statuses.tolist(), counts.tolist(), strict=True)),
            qp_status_counts=dict(zip(raw_statuses.tolist(), raw_counts.tolist(), strict=True)),
            maximum_errors=native.errors.cpu().amax(0).tolist(),
        ))
        frozen = {
            "observation": {name: value.detach().cpu().clone() for name, value in obs.items()},
            "costs": captured["costs"].cpu().clone(),
            "model_config": asdict(actor.ctbr_problem.config),
            "sqp_config": asdict(actor.sqp_config), "admm_config": asdict(actor.admm_config),
            "native_status": native.status.cpu().clone(), "native_errors": native.errors.cpu().clone(),
        }
        path = args.output / f"update-{phase:02d}-minibatch-{minibatch:02d}.pt"
        torch.save(frozen, path)
        return result

    def solve(solver, *values, **kwargs):
        if phase is None:
            return original_solve(solver, *values, **kwargs)
        return timed("mpc_solve", original_solve, solver, *values, **kwargs)

    def adjoint(ctx, upstream):
        return timed("implicit_adjoint", original_adjoint, ctx, upstream)

    def backward(tensor, *values, **kwargs):
        return timed("loss_backward", original_backward, tensor, *values, **kwargs)

    def step(optimizer, *values, **kwargs):
        return timed("optimizer_step", original_step, optimizer, *values, **kwargs)

    def update(algorithm, *values, **kwargs):
        nonlocal phase, update_count, minibatch
        update_count += 1
        phase, minibatch = update_count, 0
        events.append(dict(kind="compile_counters_before", update=phase, counters=compile_counters()))
        try:
            return timed("ppo_update_including_diagnostic_io", original_update, algorithm, *values, **kwargs)
        finally:
            events.append(dict(kind="compile_counters_after", update=phase, counters=compile_counters()))
            phase = None
            (args.output / "update-profile.json").write_text(json.dumps(events, indent=2, default=str))

    CTBRMPCActor.forward = actor_forward
    TorchMPC.solve = solve
    _ImplicitTrajectory.backward = staticmethod(adjoint)
    torch.Tensor.backward = backward
    torch.optim.Adam.step = step
    PPO.update = update
    sys.argv = [
        "acmpc.py", "train", "--task", "Isaac-ACMPC-CTBR-Direct",
        "--output", str(args.output), "--num-envs", str(args.num_envs),
        "--iterations", str(args.iterations), "agent.num_steps_per_env=16",
        "agent.algorithm.num_mini_batches=2", "agent.algorithm.num_learning_epochs=1",
    ]
    try:
        acmpc_main()
    finally:
        CTBRMPCActor.forward = original_actor
        TorchMPC.solve = original_solve
        _ImplicitTrajectory.backward = staticmethod(original_adjoint)
        torch.Tensor.backward = original_backward
        torch.optim.Adam.step = original_step
        PPO.update = original_update
        if args.output.is_dir():
            (args.output / "update-profile.json").write_text(json.dumps(events, indent=2, default=str))


if __name__ == "__main__":
    main()
