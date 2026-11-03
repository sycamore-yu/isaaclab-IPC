"""Old B1 jerk accuracy through the untouched official MPCBackend + MPC.forward.

This is an accuracy diagnostic, never an M4 result. Both KKT roles retain the
same physical problem, seed, full-plan loss and original tolerances. Select an
IR=2 candidate only with explicit command-local PYTHONPATH after env.sh.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from check_acceptance import Baseline, json_safe, load_example, sha256, verify_original

ROOT = Path(__file__).resolve().parents[2]
LIBRARY = "libcudss_sparse_kkt_ffi.so"


def options():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kkt-role", choices=("original", "ir2-candidate"), required=True)
    parser.add_argument("--reference", type=Path, default=ROOT / "build/reference")
    parser.add_argument("--manifest", type=Path, default=ROOT / "build/turbompc/kkt-reference-fix/cu128/manifest.json")
    parser.add_argument(
        "--native-reference", type=Path, default=ROOT / "results/turbompc-port/sweep-linear-cu128-v1/torch-b1.npz"
    )
    parser.add_argument("--output", type=Path, help="New JSON report; raw arrays are saved beside it as .npz")
    parser.add_argument(
        "--preflight-only", action="store_true", help="Verify files/inputs on CPU without importing JAX/Torch"
    )
    args = parser.parse_args()
    if not args.preflight_only and args.output is None:
        parser.error("--output is required except for CPU preflight")
    if args.output is not None and args.output.suffix != ".json":
        parser.error("--output must end in .json")
    return args


def preflight(args):
    if os.environ.get("TURBOMPC_CUDA_PROFILE") != "cu128":
        raise RuntimeError("Run with scripts/turbompc/env.sh cu128")
    if any(importlib.metadata.version(name) != "0.6.2" for name in ("jax", "jaxlib")):
        raise RuntimeError("The pinned JAX/jaxlib 0.6.2 environment is required")
    manifest = json.loads(args.manifest.read_text())
    installed = Path(importlib.metadata.distribution("turbompc").locate_file("turbompc")).resolve()
    baseline = installed / "solvers/_ffi" / LIBRARY
    before = sha256(baseline)
    if baseline != Path(manifest["installed_baseline_kkt"]).resolve():
        raise RuntimeError("Installed baseline path differs from the build manifest")
    if (
        before != manifest["installed_baseline_kkt_sha256_before"]
        or before != manifest["installed_baseline_kkt_sha256_after"]
    ):
        raise RuntimeError("Installed baseline KKT hash differs from the preserved original")
    candidate_source = args.manifest.resolve().parent / "source"
    if args.kkt_role == "original":
        package, expected, expected_hash = installed, baseline, before
    else:
        if manifest["kkt_ir_steps"] != 2 or manifest["profile"] != "cu128":
            raise RuntimeError("Manifest does not describe the isolated cu128 IR=2 candidate")
        paths = [Path(p).resolve() for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
        if not paths or paths[0] != candidate_source:
            raise RuntimeError("Set the candidate source as PYTHONPATH AFTER env.sh")
        package = candidate_source / "turbompc"
        expected = package / "solvers/_ffi" / LIBRARY
        expected_hash = manifest["library_sha256"]
        if expected != Path(manifest["library"]).resolve() or expected == baseline:
            raise RuntimeError("Candidate library location is inconsistent")
    spec = importlib.util.find_spec("turbompc")
    if spec is None or Path(spec.origin).resolve() != package / "__init__.py":
        raise RuntimeError(f"Python package does not match --kkt-role {args.kkt_role}")
    if sha256(expected) != expected_hash:
        raise RuntimeError("Selected KKT library SHA256 differs from the manifest")

    sys.path.insert(0, str(ROOT / "scripts"))
    from turbompc_benchmark_common import UPSTREAM_COMMIT, make_case, physical_residuals

    revision = subprocess.check_output(["git", "-C", str(args.reference), "rev-parse", "HEAD"], text=True).strip()
    if revision != UPSTREAM_COMMIT or manifest["upstream_commit"] != UPSTREAM_COMMIT:
        raise RuntimeError("Reference/candidate commit differs from the fixed old benchmark")
    verified = verify_original(args.reference, args.reference, package)
    case = make_case(1, 15, False)
    with np.load(args.native_reference, allow_pickle=False) as saved:
        for key, value in case.items():
            if f"input_{key}" not in saved or not np.array_equal(saved[f"input_{key}"], value):
                raise RuntimeError(f"The saved native input_{key} differs from the old fixed B1 case")
        native = {key: saved[key].copy() for key in ("trajectory", "cost_gradient", "state_gradient", "status")}
    for key, shape in (("trajectory", (1, 16, 12)), ("cost_gradient", (1, 12)), ("state_gradient", (1, 9))):
        if native[key].shape != shape or native[key].dtype != np.float64 or not np.isfinite(native[key]).all():
            raise RuntimeError(f"Invalid saved native {key}")
    if not np.all(native["status"] == 0):
        raise RuntimeError("Saved native B1 was not solved")
    provenance = {
        "upstream_commit": revision,
        "kkt_role": args.kkt_role,
        "actual_python_package_selected": str(package),
        "verified_upstream_files": verified,
        "expected_kkt_library": str(expected),
        "expected_kkt_library_sha256": expected_hash,
        "installed_baseline_library": str(baseline),
        "installed_baseline_sha256_before": before,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "native_reference": str(args.native_reference.resolve()),
        "native_reference_sha256": sha256(args.native_reference),
        "same_saved_inputs": True,
        "script_sha256": sha256(Path(__file__)),
        "legacy_case_helper_sha256": sha256(ROOT / "scripts/turbompc_benchmark_common.py"),
        "legacy_benchmark_helper_sha256": sha256(ROOT / "scripts/benchmark_turbompc.py"),
    }
    return SimpleNamespace(
        case=case,
        native=native,
        package=package,
        expected=expected,
        expected_hash=expected_hash,
        baseline=baseline,
        before=before,
        provenance=provenance,
        physical=physical_residuals,
    )


def relative_error(value, reference):
    return {
        "relative_l2": float(np.linalg.norm(value - reference) / max(float(np.linalg.norm(reference)), 1e-12)),
        "max_abs": float(np.max(np.abs(value - reference))),
        "reference_norm": float(np.linalg.norm(reference)),
        "finite": bool(np.isfinite(value).all() and np.isfinite(reference).all()),
    }


class OfficialJerk:
    """Problem adapter only; backend and autograd bridge are loaded unchanged."""

    def __init__(self, case, source):
        import jax
        import jax.numpy as jnp
        import torch
        from turbompc.dynamics.base_dynamics import Dynamics
        from turbompc.problems.optimal_control_problem import OptimalControlProblem
        from turbompc.utils.load_params import load_problem_params, load_solver_params

        backend_module = load_example("_jerk_official_backend", source / "examples/rl/util/mpc.py")
        bridge_module = load_example("_jerk_official_bridge", source / "examples/rl/util/jax_torch.py")

        class Model(Dynamics):
            def state_dot(self, x, u, params):
                return jnp.concatenate((x[3:6], x[6:9], u))

        class Problem:
            N, nx, nu, cost_dim = 15, 9, 3, 12
            constrain_initial_control = False

            def __init__(self):
                self.dynamics = Model(
                    {
                        "num_states": 9,
                        "num_controls": 3,
                        "names_states": [str(i) for i in range(9)],
                        "names_controls": [str(i) for i in range(3)],
                    }
                )
                self.diagonal = jnp.asarray(case["diagonal"])
                self.base = load_problem_params("required_parameters.yaml")
                self.base.update(
                    horizon=15,
                    discretization_resolution=case["dt"],
                    discretization_scheme=2,
                    initial_state=jnp.asarray(case["x0"][0]),
                    initial_guess_final_state=jnp.asarray(case["reference"]),
                    initial_control=jnp.zeros(3),
                    constrain_initial_control=False,
                    rescale_optimization_variables=False,
                    state_rescaling_min=jnp.zeros(9),
                    state_rescaling_max=jnp.ones(9),
                    control_rescaling_min=jnp.zeros(3),
                    control_rescaling_max=jnp.ones(3),
                    reference_state_trajectory=jnp.tile(jnp.asarray(case["reference"])[None], (16, 1)),
                    reference_control_trajectory=jnp.zeros((16, 3)),
                    weights_penalization_reference_state_trajectory=self.diagonal[:9] / 2,
                    weights_penalization_final_state=jnp.zeros(9),
                    weights_penalization_control_squared=self.diagonal[9:] / 2,
                    weights_penalization_control_rate=jnp.zeros(3),
                    state_min_bounds=jnp.full(9, -case["state_limit"]),
                    state_max_bounds=jnp.full(9, case["state_limit"]),
                    control_min_bounds=jnp.full(3, -case["control_limit"]),
                    control_max_bounds=jnp.full(3, case["control_limit"]),
                    use_slack_variables=False,
                    dynamics_state_dot_params={},
                )

            def make_ocp(self):
                return OptimalControlProblem(self.dynamics, self.base)

            def params_from_context(self, context):
                return {**self.base, "initial_state": context}, None

            def cost_weights(self, costs, context=None):
                # Context is differentiated once through problem_params. Adding
                # it to weights too would duplicate the upstream initial VJP.
                weights = self.diagonal * jnp.exp(costs) / 2
                return {
                    "weights_penalization_reference_state_trajectory": weights[..., :9],
                    "weights_penalization_control_squared": weights[..., 9:],
                }

            @staticmethod
            def clip_control(control):
                return jnp.clip(control, -case["control_limit"], case["control_limit"])

            @staticmethod
            def mpc_warm_start_state(predicted_state, context):
                return context

            def build_mpc_inputs(self, costs, context):
                states = jnp.broadcast_to(context[:, None], (context.shape[0], 16, 9))
                controls = jnp.zeros((context.shape[0], 16, 3))
                return self.params_from_context(context)[0], self.cost_weights(costs, context), states, controls

        parameters = load_solver_params("turbompc.yaml")
        parameters.update(
            num_sqp_iteration_max=15,
            tol_convergence=1e-6,
            convergence_criterion="first_order",
            constraint_handling="projection",
            log_barrier_kappa=0.0,
        )
        parameters["linesearch"]["method"] = "merit"
        parameters["admm"].update(max_iter=4000, eps_abs=1e-8, eps_rel=1e-8)
        self.problem = Problem()
        self.backend = backend_module.MPCBackend(
            {"solver": {**parameters, "forward_backend": "admm_fused_cudss", "backward_backend": "direct_cudss_ffi"}},
            self.problem,
        )
        self.bridge = bridge_module.MPC(self.backend)
        self.seed = backend_module.TurboSeed(
            jnp.asarray(case["initial"][..., :9]), jnp.asarray(case["initial"][..., 9:]), jnp.ones(1, dtype=bool)
        )
        self.costs = torch.tensor(case["costs"], device="cuda:0", dtype=torch.float64, requires_grad=True)
        self.context = torch.tensor(case["x0"], device="cuda:0", dtype=torch.float64, requires_grad=True)
        self.probe = torch.tensor(case["probe"], device="cuda:0", dtype=torch.float64)
        self.raw_vjp = jax.jit(jax.vmap(lambda pb, dx, du: pb((dx, du))))
        self.implementation = {
            "backend_class": f"{type(self.backend).__module__}.{type(self.backend).__name__}",
            "bridge_class": f"{type(self.bridge).__module__}.{type(self.bridge).__name__}",
            "backend_file": str(Path(backend_module.__file__).resolve()),
            "bridge_file": str(Path(bridge_module.__file__).resolve()),
            "entry_point": "official MPC.forward (mean, full controls)",
            "context_route": "params_from_context.initial_state only; one initial-state VJP",
            "effective_solver_parameters": self.backend.solver.params,
        }

    def forward(self, costs=None, context=None, capture=False):
        callbacks = {}

        def callback(name, value):
            callbacks[name] = value.detach().cpu().numpy().copy()

        mean, controls = self.bridge.forward(
            self.costs if costs is None else costs,
            self.context if context is None else context,
            self.seed,
            grad_callback=callback if capture else None,
        )
        return mean, controls, self.bridge.pop_result(), callbacks

    def snapshot(self, mean, controls, result):
        states, raw_controls = np.asarray(result.iterate.states), np.asarray(result.iterate.controls)
        return {
            "trajectory": np.concatenate((states, raw_controls), axis=-1),
            "returned_controls": controls.detach().cpu().numpy(),
            "mean": mean.detach().cpu().numpy(),
            "status": np.asarray(result.stats["status"]),
            "valid": np.asarray(result.valid),
            "convergence": np.asarray(result.stats["sqp_convergence_error"]),
            "iterations": np.asarray(result.stats["sqp_iterations"]),
        }

    def active_masks(self, result, snapshot):
        """Replay the identical official forward step to inspect its KKT state.

        MPCResult exposes only TurboSeed. This extra unchanged official step
        returns TurboMPCSolution; its primal plan must match MPC.forward's plan.
        """
        import jax
        import jax.numpy as jnp

        cache = result.cache
        sharding = cache.raw_cost.sharding

        def put(value):
            return jax.device_put(value, sharding)

        values = (
            cache.raw_cost,
            cache.context,
            put(self.seed.states),
            put(self.seed.controls),
            put(self.seed.valid),
            put(self.seed.valid),
            put(jnp.ones(1, dtype=bool)),
        )
        solution = self.backend._step_for(1)(*values, retain=False)[0]
        trajectory = np.concatenate((np.asarray(solution.states), np.asarray(solution.controls)), axis=-1)
        kkt = solution.kkt_state
        return {
            "lower": np.asarray(kkt.ineq_active_lower_idx),
            "upper": np.asarray(kkt.ineq_active_upper_idx),
            "dual": np.asarray(kkt.y_ineq),
            "replay_plan_max_abs": float(np.max(np.abs(trajectory - snapshot["trajectory"]))),
            "replay_status": np.asarray(solution.status),
        }

    def close(self):
        # Reuse the existing helper's FFI-owner cleanup only, not its backend.
        Baseline.close(self)


def snapshot_report(snapshot, case, physical):
    residuals = physical(snapshot["trajectory"], case, False)
    return {
        "status": snapshot["status"],
        "valid": snapshot["valid"],
        "convergence": snapshot["convergence"],
        "solved": bool(
            np.all(snapshot["status"] == 0) and np.all(snapshot["valid"]) and np.all(snapshot["convergence"] <= 1e-6)
        ),
        "finite": all(bool(np.isfinite(value).all()) for value in snapshot.values()),
        "physical_residuals": residuals,
        "physical_maximum": max(residuals.values()),
        "returned_plan_max_abs": float(np.max(np.abs(snapshot["returned_controls"] - snapshot["trajectory"][..., 9:]))),
    }


def finite_differences(runtime, base, base_masks, inputs, arrays):
    import torch

    result = {
        "method": "all cost/state coordinates and original seed45 cost direction; full-plan weighted loss",
        "active_mask_source": "unmodified MPCBackend._step_for retain=False KKT state, matched to bridge plan",
        "steps": [],
        "perturbations": [],
    }
    direction = np.random.default_rng(45).normal(size=(1, 12))
    direction /= np.linalg.norm(direction)
    arrays["old_cost_direction"] = direction

    def evaluate(costs, context, label, step, variable, index, sign):
        with torch.no_grad():
            mean, controls, solution, _ = runtime.forward(
                torch.as_tensor(costs, device="cuda:0", dtype=torch.float64),
                torch.as_tensor(context, device="cuda:0", dtype=torch.float64),
            )
            loss = float((controls * runtime.probe).sum().cpu())
        snapshot = runtime.snapshot(mean, controls, solution)
        masks = runtime.active_masks(solution, snapshot)
        metrics = snapshot_report(snapshot, {**inputs.case, "x0": context}, inputs.physical)
        metrics.update(
            label=label,
            step=step,
            variable=variable,
            coordinate=index,
            sign=sign,
            loss=loss,
            active_masks_equal=bool(
                np.array_equal(masks["lower"], base_masks["lower"])
                and np.array_equal(masks["upper"], base_masks["upper"])
            ),
            mask_replay_plan_max_abs=masks["replay_plan_max_abs"],
            mask_replay_solved=bool(np.all(masks["replay_status"] == 0)),
            mask_duals_finite=bool(np.isfinite(masks["dual"]).all()),
        )
        result["perturbations"].append(metrics)
        arrays.update({f"{label}_{key}": value for key, value in snapshot.items()})
        arrays.update({f"{label}_active_{key}": value for key, value in masks.items()})
        arrays[f"{label}_input_costs"], arrays[f"{label}_input_x0"] = costs, context
        return loss

    for step_index, step in enumerate((1e-4, 5e-5)):
        entry = {"step": step, "coordinate_gradients": {}}
        for variable, input_key, gradient_key in (
            ("cost", "costs", "cost_gradient"),
            ("state", "x0", "state_gradient"),
        ):
            numerical = np.empty_like(inputs.case[input_key])
            for index in range(numerical.shape[-1]):
                losses = []
                for sign, suffix in ((1.0, "plus"), (-1.0, "minus")):
                    costs, context = inputs.case["costs"].copy(), inputs.case["x0"].copy()
                    (costs if variable == "cost" else context)[0, index] += sign * step
                    label = f"fd_h{step_index}_{variable}_{index}_{suffix}"
                    losses.append(evaluate(costs, context, label, step, variable, index, sign))
                numerical[0, index] = (losses[0] - losses[1]) / (2 * step)
            arrays[f"fd_h{step_index}_{gradient_key}"] = numerical
            entry["coordinate_gradients"][gradient_key] = {
                "official_vs_fd": relative_error(base[gradient_key], numerical),
                "native_vs_fd": relative_error(inputs.native[gradient_key], numerical),
            }
        losses = []
        for sign, suffix in ((1.0, "plus"), (-1.0, "minus")):
            label = f"fd_h{step_index}_old_direction_{suffix}"
            losses.append(
                evaluate(
                    inputs.case["costs"] + sign * step * direction,
                    inputs.case["x0"].copy(),
                    label,
                    step,
                    "old_cost_direction",
                    None,
                    sign,
                )
            )
        numerical = (losses[0] - losses[1]) / (2 * step)
        analytic = float(np.sum(base["cost_gradient"] * direction))
        native_analytic = float(np.sum(inputs.native["cost_gradient"] * direction))
        entry["old_direction"] = {
            "seed": 45,
            "official_analytic": analytic,
            "native_analytic": native_analytic,
            "numerical": numerical,
            "official_relative_error": abs(analytic - numerical) / max(abs(numerical), 1e-12),
            "native_relative_error": abs(native_analytic - numerical) / max(abs(numerical), 1e-12),
        }
        result["steps"].append(entry)
    rows = result["perturbations"]
    result.update(
        forward_perturbations=len(rows),
        all_solved=all(row["solved"] and row["mask_replay_solved"] for row in rows),
        all_finite=all(row["finite"] and row["mask_duals_finite"] and np.isfinite(row["loss"]) for row in rows),
        all_active_masks_equal=all(row["active_masks_equal"] for row in rows),
        all_mask_replay_plans_match=all(row["mask_replay_plan_max_abs"] <= 1e-8 for row in rows),
        physical_maximum=max(row["physical_maximum"] for row in rows),
    )
    return result


def run(args, inputs, report, arrays):
    import jax
    import jax.numpy as jnp
    import torch
    from turbompc.solvers.backward import backward_kkt_cudss_ffi as kkt

    loaded = Path(kkt._LIB._name).resolve()
    if loaded != inputs.expected or sha256(loaded) != inputs.expected_hash:
        raise RuntimeError(f"Actually loaded KKT library does not match the explicit role: {loaded}")
    if not Path(kkt.__file__).resolve().is_relative_to(inputs.package):
        raise RuntimeError("Actually imported KKT Python source does not match the selected package")
    report["provenance"].update(actual_loaded_kkt_library=str(loaded), actual_loaded_kkt_sha256=sha256(loaded))
    jax.config.update("jax_enable_x64", True)
    report["versions"] = {"jax": jax.__version__, "torch": torch.__version__, "torch_cuda": torch.version.cuda}
    runtime = OfficialJerk(inputs.case, args.reference)
    try:
        report["implementation"] = runtime.implementation
        mean, controls, solution, callbacks = runtime.forward(capture=True)
        gradients = torch.autograd.grad((controls * runtime.probe).sum(), (runtime.costs, runtime.context))
        base = runtime.snapshot(mean, controls, solution)
        base.update(
            cost_gradient=gradients[0].detach().cpu().numpy(), state_gradient=gradients[1].detach().cpu().numpy()
        )
        dc, dx = runtime.raw_vjp(
            solution.cache.pullbacks, jnp.zeros_like(solution.states), jnp.asarray(inputs.case["probe"])
        )
        raw = {"cost_gradient": np.asarray(dc), "state_gradient": np.asarray(dx)}
        masks = runtime.active_masks(solution, base)
        report["base"] = snapshot_report(base, inputs.case, inputs.physical)
        report["base"].update(
            trajectory_max_abs_vs_native=float(np.max(np.abs(base["trajectory"] - inputs.native["trajectory"]))),
            controls_max_abs_vs_native=float(
                np.max(np.abs(base["returned_controls"] - inputs.native["trajectory"][..., 9:]))
            ),
            gradients_vs_native={key: relative_error(base[key], inputs.native[key]) for key in raw},
            active_lower_count=int(masks["lower"].sum()),
            active_upper_count=int(masks["upper"].sum()),
            mask_replay_plan_max_abs=masks["replay_plan_max_abs"],
            mask_replay_status=masks["replay_status"],
            mask_duals_finite=bool(np.isfinite(masks["dual"]).all()),
        )
        callback_keys = {"mpc_cost_adjoint", "mpc_context_adjoint"}
        finite = set(callbacks) == callback_keys and all(
            np.isfinite(v).all() for v in (*raw.values(), *callbacks.values())
        )
        report["retained_gradients"] = {
            "raw_vjp_and_callbacks_finite": bool(finite),
            "callback_names": sorted(callbacks),
            "raw_vs_official_bridge": {key: relative_error(raw[key], base[key]) for key in raw},
            "raw_vs_native": {key: relative_error(raw[key], inputs.native[key]) for key in raw},
        }
        arrays.update({f"base_{key}": value for key, value in base.items()})
        arrays.update({f"base_active_{key}": value for key, value in masks.items()})
        arrays.update({f"raw_retained_{key}": value for key, value in raw.items()})
        arrays.update({f"callback_{key}": value for key, value in callbacks.items()})
        report["finite_difference"] = finite_differences(runtime, base, masks, inputs, arrays)
        fd = report["finite_difference"]
        report["checks"] = {
            "base_solved_finite": report["base"]["solved"] and report["base"]["finite"],
            "raw_vjp_and_callbacks_finite": bool(finite),
            "full_controls_vs_native": report["base"]["controls_max_abs_vs_native"] <= 1e-4,
            "cost_gradient_vs_native": report["base"]["gradients_vs_native"]["cost_gradient"]["relative_l2"] <= 1e-3,
            "state_gradient_vs_native": report["base"]["gradients_vs_native"]["state_gradient"]["relative_l2"] <= 1e-3,
            "base_and_perturbations_physically_feasible": max(
                report["base"]["physical_maximum"], fd["physical_maximum"]
            )
            <= 1e-5,
            "perturbations_solved_finite": fd["all_solved"] and fd["all_finite"],
            "stable_active_masks": fd["all_active_masks_equal"]
            and fd["all_mask_replay_plans_match"]
            and masks["replay_plan_max_abs"] <= 1e-8
            and bool(np.all(masks["replay_status"] == 0))
            and bool(np.isfinite(masks["dual"]).all()),
            "official_full_gradients_vs_both_fd_steps": all(
                value["official_vs_fd"]["finite"] and value["official_vs_fd"]["relative_l2"] <= 1e-3
                for step in fd["steps"]
                for value in step["coordinate_gradients"].values()
            ),
            "official_old_cost_direction_vs_both_fd_steps": all(
                step["old_direction"]["official_relative_error"] <= 1e-3 for step in fd["steps"]
            ),
        }
    finally:
        runtime.close()


def main():
    args = options()
    if args.output is not None and (args.output.exists() or args.output.with_suffix(".npz").exists()):
        raise FileExistsError("Use a new output stem; existing results are preserved")
    inputs = preflight(args)
    if args.preflight_only:
        if sha256(inputs.baseline) != inputs.before:
            raise RuntimeError("Original installed KKT changed during CPU preflight")
        summary = {k: v for k, v in inputs.provenance.items() if k != "verified_upstream_files"}
        summary["verified_upstream_file_count"] = len(inputs.provenance["verified_upstream_files"])
        print(json.dumps({"cpu_preflight_only": True, "gpu_started": False, "provenance": summary}, indent=2))
        return 0
    report = {
        "role": f"official full-plan jerk accuracy diagnostic: {args.kkt_role}",
        "m4_result": False,
        "timings_collected": False,
        "case": {
            "batch": 1,
            "horizon": 15,
            "nx": 9,
            "nu": 3,
            "dtype": "float64",
            "seed": 719,
            "loss": "sum(all 16 control nodes * original saved probe)",
        },
        "provenance": inputs.provenance,
        "checks": {},
    }
    arrays = {f"input_{key}": value for key, value in inputs.case.items()}
    arrays.update({f"native_{key}": value for key, value in inputs.native.items()})
    try:
        run(args, inputs, report, arrays)
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        traceback.print_exc()
    finally:
        after = sha256(inputs.baseline)
        report["provenance"]["installed_baseline_sha256_after"] = after
        report["checks"]["installed_baseline_unchanged"] = after == inputs.before
        report["diagnostic_checks_passed"] = "error" not in report and all(report["checks"].values())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.with_suffix(".npz").open("xb") as output:
            np.savez(output, **arrays)
        with args.output.open("x") as output:
            json.dump(json_safe(report), output, indent=2, allow_nan=False)
            output.write("\n")
        summary = {
            key: value
            for key, value in report.items()
            if key not in {"provenance", "implementation", "finite_difference"}
        }
        summary["finite_difference_summary"] = {
            key: value for key, value in report.get("finite_difference", {}).items() if key != "perturbations"
        }
        print(json.dumps(json_safe(summary), indent=2, allow_nan=False))
    return int(not report["diagnostic_checks_passed"])


if __name__ == "__main__":
    raise SystemExit(main())
