"""Measure the production 28N CTBR cost map, CUDA SQP and implicit gradient.

Run each (cost_nodes, batch) pair in its own process. Compilation/cold-start is
reported separately; steady-state timings include explicit CUDA completion.
The saved result is a solver benchmark, not a navigation success claim.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from isaaclab_ipc.turbompc import ADMMConfig, SQPConfig, TorchMPC  # noqa: E402
from isaaclab_ipc.turbompc.ctbr import CTBRConfig, CTBRQPBuilder  # noqa: E402
from isaaclab_ipc.turbompc.native import _extension  # noqa: E402


def observation(builder, batch, case, faces=128):
    nodes, model = builder.cost_nodes, builder.dynamics()
    state = builder.hover_control.new_tensor([0, 0, 1, 1, 0, 0, 0, 0.2, -0.05, 0]).expand(batch, -1).clone()
    control = builder.hover_control.expand(batch, nodes - 1, -1).clone()
    control[..., 1:] = control.new_tensor([0.03, -0.04, 0.02])
    reference = model.rollout(state, control)
    reference[..., :3] += reference.new_tensor([0.12, -0.08, 0.06])
    reference[..., 7:10] += reference.new_tensor([0.1, -0.05, 0.1])
    planes = state.new_tensor(
        [[1, 0, 0, -5], [-1, 0, 0, -5], [0, 1, 0, -5], [0, -1, 0, -5], [0, 0, 1, -5], [0, 0, -1, -5]]
    )
    planes = planes.expand(batch, nodes, -1, -1).clone()
    if case == "terminal_wall":
        if nodes < 3:
            raise ValueError("The first Euler position is fixed; an active position wall needs N >= 3")
        reference[..., 2] = 1.25
        planes[:, -1, 4, 3] = -1.012
    mask = torch.ones(planes.shape[:-1], dtype=torch.bool, device=state.device)
    planes = torch.nn.functional.pad(planes, (0, 0, 0, faces - 6))
    mask = torch.nn.functional.pad(mask, (0, faces - 6))
    return {
        "ctbr_state": state,
        "ctbr_reference": reference,
        "planes": planes,
        "plane_mask": mask,
        "ctbr_warm_start": builder.hover_control.expand(batch, nodes - 1, -1).clone(),
    }


def summary(values):
    return {
        "p50": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def gpu_snapshot():
    return subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader",
        ],
        text=True,
        timeout=10,
    ).strip()


def source_hashes():
    names = ("ctbr.py", "quadrotor.py", "dynamics.py", "solver.py", "native.py", "adjoint.py")
    return {
        **{name: hashlib.sha256((ROOT / "src/isaaclab_ipc/turbompc" / name).read_bytes()).hexdigest() for name in names},
        "rl/ctbr_actor.py": hashlib.sha256((ROOT / "src/isaaclab_ipc/rl/ctbr_actor.py").read_bytes()).hexdigest(),
    }


def native_binary():
    path = Path(_extension().__file__).resolve()
    manifest = path.parent / "build-manifest.json"
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "build_manifest": str(manifest) if manifest.exists() else None,
    }


def run(args):
    torch.set_num_threads(1)
    builder = CTBRQPBuilder(CTBRConfig(cost_nodes=args.nodes), compile_kernels=args.compile).to("cuda:0")
    model = builder.dynamics()
    sqp = SQPConfig(
        max_iter=args.sqp_iterations, tolerance=args.sqp_tolerance,
        linesearch=getattr(args, "sqp_linesearch", "merit"), compile_kernels=args.compile,
    )
    admm = ADMMConfig(
        max_iter=args.qp_iterations,
        eps_abs=args.admm_tolerance,
        eps_rel=0.0,
        check_termination_every=args.check_termination_every,
        linear_solver=getattr(args, "linear_solver", "cudss_vector"),
    )
    solver = TorchMPC(
        model, sqp, admm, constraints=builder.path_constraints() if args.acceleration_constraints else None
    )
    obs = observation(builder, args.batch, args.case, args.faces)
    costs = torch.linspace(-0.025, 0.025, builder.config.cost_dim, device="cuda:0", dtype=torch.float64)
    costs = costs.expand(args.batch, -1).clone()
    weights = costs.new_tensor([0.3, -0.2, 0.1, 0.05])
    report = {
        "kind": "ctbr_28n_native_solver",
        "cost_nodes": args.nodes,
        "physical_transitions": args.nodes - 1,
        "batch": args.batch,
        "case": args.case,
        "sfc_face_capacity": args.faces,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "model": asdict(builder.config),
        "sqp": asdict(sqp),
        "admm": asdict(admm),
        "acceleration_constraints": args.acceleration_constraints,
        "gpu_before": gpu_snapshot(),
        "sources": source_hashes(),
        "warmup_count": args.warmup,
        "repeat_count": args.repeats,
        "samples": [],
        "passed": False,
    }
    arrays = {f"obs_{name}": value.detach().cpu().numpy() for name, value in obs.items()}
    arrays["cost_logits"] = costs.cpu().numpy()
    try:
        for iteration in range(args.warmup + args.repeats):
            learned = costs.detach().clone().requires_grad_(True)
            torch.cuda.synchronize()
            start = time.perf_counter()
            inputs, guess = builder(obs, learned)
            torch.cuda.synchronize()
            built = time.perf_counter()
            result = solver.solve(inputs, guess)
            torch.cuda.synchronize()
            solved = time.perf_counter()
            loss = (result.controls[:, 0] * weights).sum() / args.batch
            gradient = torch.autograd.grad(loss, learned)[0]
            torch.cuda.synchronize()
            end = time.perf_counter()
            record = {
                "iteration": iteration,
                "warmup": iteration < args.warmup,
                "build_ms": 1000 * (built - start),
                "solve_ms": 1000 * (solved - built),
                "forward_ms": 1000 * (solved - start),
                "backward_ms": 1000 * (end - solved),
                "total_ms": 1000 * (end - start),
                "failed_count": int((result.status != 0).sum()),
                "qp_failed_count": int((result.admm.status != 0).sum()),
                "sqp_iterations_max": int(result.iterations.max()),
                "admm_iterations_max": int(result.admm.iterations.max()),
                "errors_max": result.errors.amax(0).cpu().tolist(),
                "gradient_norm": float(gradient.norm()),
                "gradient_finite": bool(torch.isfinite(gradient).all()),
            }
            report["samples"].append(record)
            print(json.dumps(record), flush=True)
        measured = [sample for sample in report["samples"] if not sample["warmup"]]
        report["timing_ms"] = {
            name: summary([row[name] for row in measured])
            for name in ("build_ms", "solve_ms", "forward_ms", "backward_ms", "total_ms")
        }
        report["cold_total_ms"] = report["samples"][0]["total_ms"]
        report["passed"] = all(
            row["failed_count"] == 0 and row["gradient_finite"] and row["gradient_norm"] > 0
            for row in report["samples"]
        )
        arrays.update({f.name: getattr(inputs, f.name).detach().cpu().numpy() for f in fields(inputs)})
        arrays.update(trajectory=result.trajectory.detach().cpu().numpy(), gradient=gradient.cpu().numpy())
        if args.finite_difference:
            generator = torch.Generator(device="cuda:0").manual_seed(104)
            comparisons = []
            for direction_id in range(4):
                direction = torch.randn(costs.shape, device=costs.device, dtype=costs.dtype, generator=generator)
                if direction_id == 0:
                    direction[:, builder.config.cost_dim // 2 :] = 0
                elif direction_id == 1:
                    direction[:, : builder.config.cost_dim // 2] = 0
                direction /= direction.norm(dim=-1, keepdim=True)
                expected_rows = (gradient * direction).sum(-1) * args.batch
                expected = expected_rows.mean()
                for epsilon in (1e-2, 5e-3):
                    losses, statuses = [], []
                    for sign in (1, -1):
                        with torch.no_grad():
                            data, initial = builder(obs, costs + sign * epsilon * direction)
                            perturbed = solver.solve(data, initial)
                        losses.append((perturbed.controls[:, 0] * weights).sum(-1))
                        statuses.append(int((perturbed.status != 0).sum()))
                    actual_rows = (losses[0] - losses[1]) / (2 * epsilon)
                    actual = actual_rows.mean()
                    absolute = float((actual - expected).abs())
                    relative = absolute / max(float(expected.abs()), 1e-12)
                    row_error = (actual_rows - expected_rows).abs()
                    row_budget = 1e-5 + 0.01 * expected_rows.abs()
                    passed = not any(statuses) and bool((row_error <= row_budget).all())
                    comparisons.append(
                        {
                            "direction": direction_id,
                            "epsilon": epsilon,
                            "autograd": float(expected),
                            "finite_difference": float(actual),
                            "absolute_error": absolute,
                            "relative_error": relative,
                            "max_row_absolute_error": float(row_error.max()),
                            "max_row_error_fraction_of_budget": float((row_error / row_budget).max()),
                            "failed_counts": statuses,
                            "passed": passed,
                        }
                    )
            report["finite_difference"] = comparisons
            report["passed"] &= all(row["passed"] for row in comparisons)
            print(json.dumps({"finite_difference": comparisons}), flush=True)
    finally:
        report["native_binary"] = native_binary()
        report["sources_unchanged"] = report["sources"] == source_hashes()
        report["passed"] &= report["sources_unchanged"]
        report["gpu_after"] = gpu_snapshot()
        solver.close()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        np.savez_compressed(args.output.with_suffix(".npz"), **arrays)
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
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--acceleration-constraints", action="store_true")
    parser.add_argument("--sqp-iterations", type=int, default=25)
    parser.add_argument("--sqp-tolerance", type=float, default=1e-4)
    parser.add_argument("--sqp-linesearch", choices=("none", "merit"), default="none")
    parser.add_argument("--qp-iterations", type=int, default=4096)
    parser.add_argument("--linear-solver", choices=("cudss_vector", "cudss_inverse_chunk"), default="cudss_vector")
    parser.add_argument("--admm-tolerance", type=float, default=1e-5)
    parser.add_argument("--check-termination-every", type=int, default=25)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--finite-difference", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.batch < 1 or args.nodes < 2 or args.faces < 6 or args.warmup < 1 or args.repeats < 1:
        parser.error("positive batch/warmup/repeats, at least two nodes and six SFC faces are required")
    raise SystemExit(0 if run(args) else 1)


if __name__ == "__main__":
    main()
