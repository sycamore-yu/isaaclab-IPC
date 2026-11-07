"""Time the actual AC-MPC actor on frozen F128 observations, without a simulator."""

import argparse
import json
import time
from pathlib import Path

import torch
from benchmark_ctbr_native import gpu_snapshot, native_binary, observation, source_hashes, summary
from tensordict import TensorDict

from isaaclab_ipc.rl.ctbr_actor import CTBRMPCActor
from isaaclab_ipc.rl.ctbr_observation import make_ctbr_observation
from isaaclab_ipc.turbompc.ctbr import CTBRConfig, CTBRQPBuilder


def run(args):
    torch.set_num_threads(1)
    cfg = CTBRConfig(cost_nodes=args.nodes)
    fixture = CTBRQPBuilder(cfg).to("cuda:0")
    values = observation(fixture, args.batch, args.case, args.faces)
    reference = values["ctbr_reference"]
    control = fixture.hover_control.expand(args.batch, args.nodes, -1)
    acceleration = fixture.dynamics().acceleration(torch.cat((reference, control), -1))
    pva = torch.cat((reference[..., :3], reference[..., 7:], acceleration), -1)
    inputs = make_ctbr_observation(
        values["ctbr_state"],
        values["ctbr_state"].new_zeros((args.batch, 3)),
        reference[:, -1, :3],
        pva,
        values["ctbr_state"].new_zeros((args.batch, 1)),
        values["planes"],
        values["plane_mask"],
        ctbr_config=cfg,
        warm_start_controls=values["ctbr_warm_start"],
    )
    obs = TensorDict(inputs, batch_size=[args.batch], device="cuda:0")
    actor = CTBRMPCActor(obs, {"actor": ["policy"]}, "actor", 4, cost_nodes=args.nodes, compile_kernels=True).to(
        "cuda:0"
    )
    actor.fixed_cost = args.fixed_cost
    actor.eval()
    weights = torch.tensor([0.3, -0.2, 0.1, 0.05], device="cuda:0")
    report = {
        "kind": "ctbr_actual_actor",
        "batch": args.batch,
        "cost_nodes": args.nodes,
        "sfc_face_capacity": args.faces,
        "fixed_cost": args.fixed_cost,
        "case": args.case,
        "warmup_count": args.warmup,
        "repeat_count": args.repeats,
        "gpu_before": gpu_snapshot(),
        "sources": source_hashes(),
        "samples": [],
        "passed": False,
    }
    try:
        for iteration in range(args.warmup + args.repeats):
            actor.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            start = time.perf_counter()
            actions = actor(obs)
            torch.cuda.synchronize()
            forward = time.perf_counter()
            if not args.fixed_cost:
                loss = (actions * weights).sum() / args.batch
                loss.backward()
            torch.cuda.synchronize()
            end = time.perf_counter()
            native = actor.last_native_result
            row = {
                "iteration": iteration,
                "warmup": iteration < args.warmup,
                "forward_ms": 1000 * (forward - start),
                "backward_ms": 1000 * (end - forward),
                "total_ms": 1000 * (end - start),
                "failed_count": int((native.status != 0).sum()),
                "errors_max": native.errors.amax(0).cpu().tolist(),
                "sqp_iterations_max": int(native.iterations.max()),
                "gradient_norm": None if args.fixed_cost else float(actor.mlp[-1].bias.grad.norm()),
            }
            report["samples"].append(row)
            print(json.dumps(row), flush=True)
        measured = [row for row in report["samples"] if not row["warmup"]]
        report["timing_ms"] = {
            name: summary([row[name] for row in measured]) for name in ("forward_ms", "backward_ms", "total_ms")
        }
        report["cold_total_ms"] = report["samples"][0]["total_ms"]
        report["solver"] = actor.solver_metadata()
        report["passed"] = all(row["failed_count"] == 0 for row in report["samples"])
    finally:
        report["native_binary"] = native_binary()
        report["sources_unchanged"] = report["sources"] == source_hashes()
        report["passed"] &= report["sources_unchanged"]
        report["gpu_after"] = gpu_snapshot()
        actor.close()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({"output": str(args.output), "passed": report["passed"], "timing_ms": report.get("timing_ms")}),
        flush=True,
    )
    return report["passed"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--nodes", type=int, default=5)
    parser.add_argument("--faces", type=int, default=128)
    parser.add_argument("--case", choices=("open", "terminal_wall"), default="open")
    parser.add_argument("--fixed-cost", action="store_true")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(0 if run(args) else 1)


if __name__ == "__main__":
    main()
