"""Interleave original bridge and native Torch calls under the same GPU load.

Both backends use the same CUDA 12 profile and identical inputs. Ordering is
randomized within each pair; neither the original solver nor bridge is edited.
This measures shared-load behavior, not an exclusive-GPU real-time guarantee.
"""

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from benchmark_turbompc import gpu_snapshot, prepare_jax, prepare_torch
from turbompc_benchmark_common import make_case, physical_residuals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--nonlinear", action="store_true")
    parser.add_argument("--pairs", type=int, default=60)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=Path("build/reference"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Preserve existing result: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    case = make_case(args.batch, args.horizon, args.nonlinear)
    settings = dict(vars(args), compile=True, backward_backend="direct_cudss_ffi")
    bridge = prepare_jax(case, SimpleNamespace(**settings, backend="bridge"))
    native = prepare_torch(case, SimpleNamespace(**settings, backend="torch"))
    runtimes = {"bridge": bridge, "torch": native}
    exports, cold = {}, {}
    for name, runtime in runtimes.items():
        t0 = time.perf_counter()
        value = runtime.full()
        runtime.ready(value)
        cold[name] = time.perf_counter() - t0
        exports[name] = runtime.export(value)
    before = gpu_snapshot()
    order = np.random.default_rng(918).integers(0, 2, size=args.pairs)
    timings = {name: {"bridge": [], "torch": []} for name in ("forward", "forward_backward")}
    for measure, method in (("forward", "forward"), ("forward_backward", "full")):
        for _ in range(5):
            for runtime in runtimes.values():
                runtime.ready(getattr(runtime, method)())
        for reverse in order:
            for name in ("torch", "bridge") if reverse else ("bridge", "torch"):
                runtime = runtimes[name]
                runtime.ready(None)
                start = time.perf_counter()
                value = getattr(runtime, method)()
                runtime.ready(value)
                timings[measure][name].append((time.perf_counter() - start) * 1000)
    statistics = {}
    rng = np.random.default_rng(726)
    for measure, sample in timings.items():
        n, b = np.asarray(sample["torch"]), np.asarray(sample["bridge"])
        indices = rng.integers(0, len(n), size=(2000, len(n)))
        bootstrap = np.median(n[indices], axis=1) / np.median(b[indices], axis=1)
        statistics[measure] = {
            "torch_median_ms": float(np.median(n)),
            "bridge_median_ms": float(np.median(b)),
            "torch_p95_ms": float(np.percentile(n, 95)),
            "bridge_p95_ms": float(np.percentile(b, 95)),
            "median_ratio": float(np.median(n) / np.median(b)),
            "p95_ratio": float(np.percentile(n, 95) / np.percentile(b, 95)),
            "median_ratio_bootstrap_95ci": np.percentile(bootstrap, [2.5, 97.5]).tolist(),
            "samples_ms": sample,
        }
    n, b = exports["torch"], exports["bridge"]
    direction = np.random.default_rng(45).normal(size=case["costs"].shape)
    direction /= np.linalg.norm(direction)
    h = 1e-4
    fd = {}
    for name, runtime in runtimes.items():
        d = runtime.tensor(direction)
        numerical = (runtime.fd(d, h) - runtime.fd(d, -h)) / (2 * h)
        analytic = float(np.sum(exports[name]["cost_gradient"] * direction))
        fd[name] = dict(
            analytic=analytic,
            numerical=numerical,
            absolute_error=abs(analytic - numerical),
            relative_error=abs(analytic - numerical) / max(abs(numerical), 1e-12),
        )
    full = statistics["forward_backward"]
    report = dict(
        batch=args.batch,
        horizon=args.horizon,
        nonlinear=args.nonlinear,
        pairs=args.pairs,
        condition="Paired randomized order; same CUDA version; externally shared GPU",
        gpu_before=before,
        gpu_after=gpu_snapshot(),
        cold_seconds=cold,
        versions={name: runtime.version for name, runtime in runtimes.items()},
        statistics=statistics,
        finite_difference=fd,
        forward_max_abs=float(np.max(np.abs(n["trajectory"] - b["trajectory"]))),
        all_solved=bool(np.all(n["status"] == 0) and np.all(b["status"] == 0)),
        residuals={
            name: physical_residuals(value["trajectory"], case, args.nonlinear) for name, value in exports.items()
        },
        measured_latency_gate=full["median_ratio"] <= 1.1 and full["p95_ratio"] <= 1.1,
        statistical_median_gate=full["median_ratio_bootstrap_95ci"][1] <= 1.1,
        memory={name: runtime.memory() for name, runtime in runtimes.items()},
    )
    for name, export in exports.items():
        np.savez(args.output.with_name(args.output.stem + "-" + name).with_suffix(".npz"), **export)
    # Cleanup is part of a successful run, although it is not timed as control work.
    native.close()
    bridge.close()
    report["cleanup_ok"] = True
    args.output.write_text(json.dumps(report, indent=2, default=lambda x: int(x)) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("statistics", "memory")}, indent=2))
    print("F+B", json.dumps({k: v for k, v in full.items() if k != "samples_ms"}))


if __name__ == "__main__":
    main()
