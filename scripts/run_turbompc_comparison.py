"""Legacy sequential diagnostics with preserved process failures.

This runner never stops other jobs or edits benchmark outputs. Use a new output
directory per sweep. Its custom bridge backend is not the official MPCBackend.
Use scripts/turbompc/check_acceptance.py for formal shared-GPU A/B acceptance.
"""

import argparse
import json
import os
import resource
import subprocess
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 16, 64, 256])
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--nonlinear", action="store_true")
    parser.add_argument("--backends", nargs="+", choices=["torch", "bridge", "jax"], default=["bridge", "torch"])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    environment = dict(
        os.environ,
        TURBOMPC_NATIVE_BUILD_DIR=str(root / "build/turbompc/native/cu128"),
        TORCHINDUCTOR_CACHE_DIR=str(root / "build/inductor-cu128"),
        TRITON_CACHE_DIR=str(root / "build/triton-cu128"),
        TORCHINDUCTOR_COMPILE_THREADS="2",
    )
    runs = []
    for batch in args.batches:
        for backend in args.backends:
            name = f"{backend}-b{batch}"
            output = (args.output / f"{name}.json").resolve()
            command = [
                "bash",
                "scripts/turbompc/env.sh",
                "cu128",
                "env",
                f"PYTHONPATH={root / 'src'}",
                "python",
                "scripts/benchmark_turbompc.py",
                "--backend",
                backend,
                "--batch",
                str(batch),
                "--repeats",
                str(args.repeats),
                "--warmup",
                "5",
                "--finite-difference",
                "--output",
                str(output),
            ]
            if backend == "torch":
                command.append("--compile")
            if args.nonlinear:
                command.append("--nonlinear")
            print("Running", name, flush=True)
            with (args.output / f"{name}.log").open("x") as log:
                process = subprocess.run(command, cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT)
            runs.append(dict(name=name, command=command, returncode=process.returncode))
            (args.output / "processes.json").write_text(json.dumps(runs, indent=2) + "\n")
            print(name, "exit", process.returncode, flush=True)

    comparisons = []
    for batch in args.batches:
        native_path, bridge_path = [args.output / f"{backend}-b{batch}.json" for backend in ("torch", "bridge")]
        if not native_path.exists() or not bridge_path.exists():
            continue
        native, bridge = json.loads(native_path.read_text()), json.loads(bridge_path.read_text())
        n, b = np.load(native_path.with_suffix(".npz")), np.load(bridge_path.with_suffix(".npz"))
        same_input = all(np.array_equal(n[k], b[k]) for k in n.files if k.startswith("input_"))
        gradient_error = float(
            np.linalg.norm(n["cost_gradient"] - b["cost_gradient"]) / max(np.linalg.norm(n["cost_gradient"]), 1e-15)
        )
        ratio50 = native["forward_backward"]["median_ms"] / bridge["forward_backward"]["median_ms"]
        ratio95 = native["forward_backward"]["p95_ms"] / bridge["forward_backward"]["p95_ms"]
        comparisons.append(
            dict(
                batch=batch,
                same_inputs=same_input,
                forward_max_abs=float(np.max(np.abs(n["trajectory"] - b["trajectory"]))),
                upstream_cost_gradient_relative_error=gradient_error,
                native_fd=native.get("finite_difference"),
                bridge_fd=bridge.get("finite_difference"),
                all_solved=bool(np.all(n["status"] == 0) and np.all(b["status"] == 0)),
                median_ratio=ratio50,
                p95_ratio=ratio95,
                latency_threshold_met=ratio50 <= 1.1 and ratio95 <= 1.1,
                native_residuals=native["physical_residuals"],
                bridge_residuals=bridge["physical_residuals"],
            )
        )
    summary = dict(
        qualification="Legacy sequential diagnostic, not M4. Shared GPU is authorized; formal acceptance uses check_acceptance.py.",
        nonlinear=args.nonlinear,
        processes=runs,
        comparisons=comparisons,
        all_processes_ok=all(run["returncode"] == 0 for run in runs),
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
