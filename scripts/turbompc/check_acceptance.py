"""Paired CUDA 12 M4 checks: untouched official Torch/JAX bridge vs TorchMPC.

Default: N=15, float64, batches 1/16/64/256, 100 AB/BA pairs per batch.
Smaller --pairs/--batches runs are diagnostics and cannot pass the complete gate.
The fixed official jerk regression is a gate; legacy custom-backend history is diagnostic only.
No upstream source or solver settings are changed to repair the A baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import threading
import time
import traceback
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REVISION = "7050a4f367d08504ecb6f5820d9161e704df5816"
N, NX, NU, DT = 15, 4, 2, 0.1
REQUIRED_BATCHES = (1, 16, 64, 256)
THRESHOLDS = {
    "u0_max_abs": 1e-4,
    "gradient_relative": 1e-3,
    "physical_violation": 1e-5,
    "latency_ratio": 1.1,
    "gradient_norm_floor": 1e-12,
    "stable_interior_margin": 1e-4,
}
ADMM = {
    "rho": 0.1,
    "sigma": 1e-6,
    "max_iter": 4000,
    "eps_abs": 1e-10,
    "eps_rel": 1e-10,
    "eps_prim_inf": 1e-4,
    "eps_dual_inf": 1e-4,
    "rho_min": 1e-6,
    "rho_max": 1e6,
    "check_termination_every": 25,
    "check_infeasibility_every": 25,
    "adapt_rho_every": 25,
    "adaptive_rho_tolerance": 5.0,
    "rho_update_max_factor": 1e6,
    "relaxation_parameter": 1.6,
    "rho_f_factor": 1000.0,
}
SOLVER = {
    "forward_backend": "admm_fused_cudss",
    "backward_backend": "direct_cudss_ffi",
    "constraint_handling": "projection",
    "log_barrier_kappa": 0.0,
    "num_sqp_iteration_max": 15,
    "tol_convergence": 1e-8,
    "convergence_criterion": "first_order",
    "linesearch": {
        "method": "merit",
        "alphas": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "merit": {
            "armijo_coefficient": 0.4,
            "penalty_min": 0.1,
            "penalty_max": 1e6,
            "constraint_norm_floor": 1e-8,
        },
    },
    "admm": ADMM,
}
STAGE_WEIGHT = np.array([1.0, 1.0, 0.2, 0.2])
FINAL_WEIGHT = np.array([2.0, 2.0, 0.2, 0.2])
CONTROL_WEIGHT = np.array([0.2, 0.3])


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def load_example(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class GPULoad:
    """Timestamped sampling in a CPU thread; never stop other GPU processes."""

    def __init__(self):
        self.samples = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        while not self.stop_event.is_set():
            sample = {"time_ns": time.time_ns()}
            try:
                for key, query in (
                    (
                        "gpus",
                        "--query-gpu=uuid,name,driver_version,memory.used,memory.total,"
                        "utilization.gpu,utilization.memory,power.draw,temperature.gpu,clocks.sm,pstate",
                    ),
                    ("compute_processes", "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory"),
                ):
                    result = subprocess.run(
                        ["nvidia-smi", query, "--format=csv,noheader,nounits"],
                        capture_output=True,
                        text=True,
                        timeout=5,
                        check=False,
                    )
                    sample[key] = {
                        "returncode": result.returncode,
                        "csv": result.stdout.strip(),
                        "stderr": result.stderr.strip(),
                    }
            except Exception as error:
                sample["error"] = f"{type(error).__name__}: {error}"
            self.samples.append(sample)
            self.stop_event.wait(1.0)

    def start(self):
        self.thread.start()

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=12)


def verify_original(source, repository, installed):
    """Compare the imported Python/YAML and both bridge files to git objects."""
    archive = subprocess.check_output(
        [
            "git",
            "-C",
            str(repository),
            "archive",
            REVISION,
            "turbompc",
            "examples/rl/util/jax_torch.py",
            "examples/rl/util/mpc.py",
        ]
    )
    checked = {}
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as bundle:
        for entry in bundle.getmembers():
            if not entry.isfile():
                continue
            relative = Path(entry.name)
            data = bundle.extractfile(entry).read()
            if relative.parts[0] == "turbompc" and relative.suffix in {".py", ".yaml"}:
                actual = installed / relative.relative_to("turbompc")
            elif entry.name in {"examples/rl/util/jax_torch.py", "examples/rl/util/mpc.py"}:
                actual = source / relative
            else:
                continue
            if not actual.is_file() or actual.read_bytes() != data:
                raise RuntimeError(f"A is not the pinned original: {actual}")
            checked[str(actual)] = hashlib.sha256(data).hexdigest()
    if not checked:
        raise RuntimeError("No upstream files were verified")
    return checked


def case_bank(batch):
    # Generate the full fixed bank before taking prefixes. No rejected rows or
    # replacement samples are selected after seeing either solver's results.
    rng = np.random.default_rng(719)
    count = max(256, batch)
    states = np.array([0.2, -0.1, 0.05, -0.02]) + rng.uniform(-0.04, 0.04, (count, NX))
    costs = rng.uniform(-0.15, 0.15, (count, 2 * NX + NU))
    seed = np.zeros((count, N + 1, NX + NU), dtype=np.float64)
    seed[..., :NX] = states[:, None]
    return {"state": states[:batch].copy(), "costs": costs[:batch].copy(), "seed": seed[:batch].copy()}


def verified_build_records():
    """Associate current source/library hashes with recorded incremental builds."""
    from record_build import file_record

    records = {}
    for role, directory in (
        ("A", ROOT / "build/turbompc/baseline/cu128"),
        ("B", ROOT / "build/turbompc/native/cu128"),
    ):
        path = directory / "build-manifest.json"
        manifest = json.loads(path.read_text())
        if manifest.get("status") != "succeeded" or not manifest.get("source_guard_passed"):
            raise RuntimeError(f"{role} has no successful guarded build record: {path}")
        if role == "A" and manifest.get("upstream_revision") != REVISION:
            raise RuntimeError("A build record is not the pinned upstream revision")
        snapshot = manifest["sources_after"]
        for name, expected in snapshot["files"].items():
            source = (
                Path(name.removeprefix("build-driver:"))
                if name.startswith("build-driver:")
                else Path(snapshot["root"]) / name
            )
            actual = file_record(
                source,
                git_format=snapshot.get("git_object_format"),
                git_mode=expected.get("expected_git_mode"),
            )
            if actual["sha256"] != expected["sha256"]:
                raise RuntimeError(f"{role} source changed since its recorded build: {source}")
        artifacts = {name: value["sha256"] for name, value in manifest["artifacts_after_call"].items()}
        for name, expected in artifacts.items():
            if sha256(Path(name)) != expected:
                raise RuntimeError(f"{role} binary changed since its recorded build: {name}")
        records[role] = {
            "manifest": str(path),
            "manifest_sha256": sha256(path),
            "build_id": manifest["build_id"],
            "incremental": manifest["incremental"],
            "scope": manifest["provenance_scope"],
            "source_files_checked": len(snapshot["files"]),
            "artifact_sha256": artifacts,
        }
    return records


def array_hash(values):
    digest = hashlib.sha256()
    for key, value in sorted(values.items()):
        digest.update(key.encode())
        digest.update(str((value.dtype.str, value.shape)).encode())
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


class Baseline:
    def __init__(self, case, source):
        import jax
        import jax.numpy as jnp
        import torch
        from turbompc.dynamics.linear_dynamics import LinearDynamics
        from turbompc.problems.optimal_control_problem import OptimalControlProblem
        from turbompc.utils.load_params import load_problem_params

        backend_module = load_example("_acceptance_official_backend", source / "examples/rl/util/mpc.py")
        bridge_module = load_example("_acceptance_official_bridge", source / "examples/rl/util/jax_torch.py")

        class LinearProblem:
            N, nx, nu, cost_dim = N, NX, NU, 2 * NX + NU
            constrain_initial_control = False
            weight_keys = (
                "weights_penalization_reference_state_trajectory",
                "weights_penalization_final_state",
                "weights_penalization_control_squared",
            )

            def __init__(self):
                self.dynamics = LinearDynamics()
                self.base = load_problem_params("linear.yaml")
                self.constant_dynamics = {k: np.asarray(v) for k, v in self.base["dynamics_state_dot_params"].items()}
                self.base.update(
                    {
                        "horizon": N,
                        "discretization_resolution": DT,
                        "discretization_scheme": 0,
                        "initial_state": jnp.zeros(NX),
                        "initial_control": jnp.zeros(NU),
                        "initial_guess_final_state": jnp.zeros(NX),
                        "constrain_initial_control": False,
                        "reference_state_trajectory": jnp.zeros((N + 1, NX)),
                        "reference_control_trajectory": jnp.zeros((N + 1, NU)),
                        self.weight_keys[0]: jnp.asarray(STAGE_WEIGHT),
                        self.weight_keys[1]: jnp.asarray(FINAL_WEIGHT),
                        self.weight_keys[2]: jnp.asarray(CONTROL_WEIGHT),
                        "weights_penalization_control_rate": jnp.zeros(NU),
                        "state_min_bounds": jnp.full(NX, -5.0),
                        "state_max_bounds": jnp.full(NX, 5.0),
                        "control_min_bounds": jnp.full(NU, -2.0),
                        "control_max_bounds": jnp.full(NU, 2.0),
                        "rescale_optimization_variables": False,
                        "use_slack_variables": False,
                        "dynamics_state_dot_params": {
                            k: jnp.broadcast_to(v, (N, *v.shape)) for k, v in self.constant_dynamics.items()
                        },
                    }
                )

            def make_ocp(self):
                return OptimalControlProblem(self.dynamics, self.base)

            def params_from_context(self, context):
                return {**self.base, "initial_state": context}, None

            def cost_weights(self, costs, context=None):
                factors = jnp.split(jnp.exp(costs), (NX, 2 * NX), axis=-1)
                return {key: self.base[key] * factor for key, factor in zip(self.weight_keys, factors, strict=True)}

            @staticmethod
            def clip_control(control):
                return jnp.clip(control, -2.0, 2.0)

            @staticmethod
            def mpc_warm_start_state(predicted_state, context):
                return context

            def build_mpc_inputs(self, costs, context):
                states = jnp.broadcast_to(context[:, None], (context.shape[0], N + 1, NX))
                controls = jnp.zeros((context.shape[0], N + 1, NU))
                return self.params_from_context(context)[0], self.cost_weights(costs), states, controls

        self.problem = LinearProblem()
        self.backend = backend_module.MPCBackend({"solver": SOLVER}, self.problem)
        self.bridge = bridge_module.MPC(self.backend)
        self.seed = backend_module.TurboSeed(
            jnp.asarray(case["seed"][..., :NX]),
            jnp.asarray(case["seed"][..., NX:]),
            jnp.ones(case["state"].shape[0], dtype=bool),
        )
        self.costs = torch.tensor(case["costs"], device="cuda:0", dtype=torch.float64, requires_grad=True)
        self.state = torch.tensor(case["state"], device="cuda:0", dtype=torch.float64, requires_grad=True)
        self.raw_vjp = jax.jit(jax.vmap(lambda pb, dx, du: pb((dx, du))))
        self.constants = self.problem.constant_dynamics

    def forward(self, costs=None, state=None, capture=False):
        callbacks = {}

        def callback(name, value):
            callbacks[name] = value.detach()

        mean = self.bridge.mean_only(
            self.costs if costs is None else costs,
            self.state if state is None else state,
            self.seed,
            grad_callback=callback if capture else None,
        )
        return mean, self.bridge.pop_result(), callbacks

    def snapshot(self, result):
        mean, solution, gradients, callbacks = result
        value = {
            "states": np.asarray(solution.iterate.states),
            "controls": np.asarray(solution.iterate.controls),
            "u0": mean.detach().cpu().numpy(),
            "status": np.asarray(solution.stats["status"]),
            "valid": np.asarray(solution.valid),
            "convergence": np.asarray(solution.stats["sqp_convergence_error"]),
            "iterations": np.asarray(solution.stats["sqp_iterations"]),
            "admm_iterations": np.asarray(solution.stats["inner_iterations"]),
        }
        if gradients is not None:
            value.update(
                cost_gradient=gradients[0].detach().cpu().numpy(), state_gradient=gradients[1].detach().cpu().numpy()
            )
        return value

    def raw_gradient_check(self, result):
        import jax.numpy as jnp

        mean, solution, _, callbacks = result
        du = jnp.zeros_like(solution.controls).at[:, 0].set(jnp.asarray(mean.detach().cpu().numpy()))
        dc, dx = self.raw_vjp(solution.cache.pullbacks, jnp.zeros_like(solution.states), du)
        arrays = {"cost_gradient": np.asarray(dc), "state_gradient": np.asarray(dx)}
        callback_arrays = {key: value.cpu().numpy() for key, value in callbacks.items()}
        valid = set(callback_arrays) == {"mpc_cost_adjoint", "mpc_context_adjoint"} and all(
            np.isfinite(value).all() for value in (*arrays.values(), *callback_arrays.values())
        )
        return {"raw_vjp_and_callbacks_finite": bool(valid), "callbacks": callback_arrays}, arrays

    def close(self):
        import jax

        jax.effects_barrier()
        solver = self.backend.solver
        for admm in (solver._admm_solver_fwd, solver._admm_solver_bwd):
            if admm is not None and admm._cudss_workspace is not None:
                admm._cudss_workspace.close()
        if solver._cudss_backward_solver is not None:
            solver._cudss_backward_solver.close()
        for schur in (solver._schur_solver_fwd, solver._schur_solver_bwd):
            if schur is not None:
                schur.close()
        jax.effects_barrier()


class Native:
    def __init__(self, case, constants, compile_kernels):
        import torch

        from isaaclab_ipc.turbompc import ADMMConfig, LinearDynamics, MPCInputs, SQPConfig, TorchMPC

        self.MPCInputs = MPCInputs
        self.tensor = lambda value: torch.as_tensor(value, device="cuda:0", dtype=torch.float64)
        matrix = np.concatenate((np.eye(NX) + DT * constants["A"], DT * constants["B"]), axis=-1)
        dynamics = LinearDynamics(self.tensor(matrix), nx=NX, offset=self.tensor(DT * constants["b"]))
        config = {k: v for k, v in ADMM.items() if k != "relaxation_parameter"}
        config["alpha"] = ADMM["relaxation_parameter"]
        self.solver = TorchMPC(
            dynamics,
            SQPConfig(
                max_iter=SOLVER["num_sqp_iteration_max"],
                tolerance=SOLVER["tol_convergence"],
                linesearch="merit",
                alphas=tuple(SOLVER["linesearch"]["alphas"]),
                armijo=0.4,
                compile_kernels=compile_kernels,
            ),
            ADMMConfig(**config),
        )
        self.costs = self.tensor(case["costs"]).requires_grad_()
        self.state = self.tensor(case["state"]).requires_grad_()
        self.seed = self.tensor(case["seed"])
        batch = self.state.shape[0]
        self.g = torch.eye(NX + NU, device="cuda:0", dtype=torch.float64).expand(batch, N + 1, -1, -1)
        self.e = torch.zeros((batch, N, NX + NU, NX + NU), device="cuda:0", dtype=torch.float64)
        self.bound = self.tensor([5.0] * NX + [2.0] * NU).expand(batch, N + 1, -1)
        self.weights = tuple(self.tensor(v) for v in (STAGE_WEIGHT, FINAL_WEIGHT, CONTROL_WEIGHT))

    def forward(self, costs=None, state=None, capture=False):
        import torch

        costs = self.costs if costs is None else costs
        state = self.state if state is None else state
        stage = self.weights[0] * costs[:, :NX].exp()
        final = self.weights[1] * costs[:, NX : 2 * NX].exp()
        control = self.weights[2] * costs[:, 2 * NX :].exp()
        stage_diagonal = 2 * torch.cat((stage, control), dim=-1)
        terminal_diagonal = 2 * torch.cat((stage + final, control), dim=-1)
        diagonal = torch.cat((stage_diagonal[:, None].expand(-1, N, -1), terminal_diagonal[:, None]), dim=1)
        inputs = self.MPCInputs(
            state,
            torch.diag_embed(diagonal),
            self.e,
            torch.zeros_like(diagonal),
            self.g,
            -self.bound,
            self.bound,
        )
        # The official MPCBackend overwrites the seed's first state from context.
        # Mirror that rule for initial-state finite differences as well.
        seed = self.seed.detach().clone()
        seed[:, 0, :NX] = state.detach()
        result = self.solver.solve(inputs, seed)
        return result.controls[:, 0], result, {}

    def snapshot(self, result):
        mean, solution, gradients, callbacks = result
        value = {
            "states": solution.states.detach().cpu().numpy(),
            "controls": solution.controls.detach().cpu().numpy(),
            "u0": mean.detach().cpu().numpy(),
            "status": solution.status.cpu().numpy(),
            "valid": (solution.status == 0).cpu().numpy(),
            "convergence": solution.errors.amax(-1).cpu().numpy(),
            "iterations": solution.iterations.cpu().numpy(),
            "admm_iterations": solution.admm.iterations.cpu().numpy(),
            "admm_status": solution.admm.status.cpu().numpy(),
            "admm_residuals": solution.admm.residuals.cpu().numpy(),
        }
        if gradients is not None:
            value.update(
                cost_gradient=gradients[0].detach().cpu().numpy(), state_gradient=gradients[1].detach().cpu().numpy()
            )
        return value

    def close(self):
        self.solver.close()


def full(runtime, capture=False):
    import torch

    mean, solution, callbacks = runtime.forward(capture=capture)
    loss = 0.5 * mean.square().sum()
    gradients = torch.autograd.grad(loss, (runtime.costs, runtime.state))
    return mean, solution, gradients, callbacks


def physical(snapshot, case, constants):
    states, controls = snapshot["states"], snapshot["controls"]
    rhs = states[:, :-1] @ constants["A"].T + controls[:, :-1] @ constants["B"].T + constants["b"]
    values = {
        "initial_state": np.max(np.abs(states[:, 0] - case["state"]), axis=-1),
        "dynamics": np.max(np.abs(states[:, 1:] - states[:, :-1] - DT * rhs), axis=(1, 2)),
        "state_bounds": np.maximum(np.abs(states) - 5.0, 0.0).max(axis=(1, 2)),
        "control_bounds": np.maximum(np.abs(controls) - 2.0, 0.0).max(axis=(1, 2)),
    }
    margin = np.minimum((5.0 - np.abs(states)).min(axis=(1, 2)), (2.0 - np.abs(controls)).min(axis=(1, 2)))
    return {
        "per_sample": values,
        "maximum": float(np.max(np.stack(list(values.values())))),
        "interior_margin_per_sample": margin,
        "finite": bool(all(np.isfinite(v).all() for v in (*values.values(), margin))),
    }


def gradient_error(value, reference):
    difference = np.linalg.norm(value - reference, axis=-1)
    norm = np.linalg.norm(reference, axis=-1)
    relative = difference / np.maximum(norm, THRESHOLDS["gradient_norm_floor"])
    finite = bool(np.isfinite(value).all() and np.isfinite(reference).all())
    return {
        "per_sample_relative": relative,
        "maximum_relative": float(np.max(relative)),
        "global_relative": float(
            np.linalg.norm(value - reference) / max(float(np.linalg.norm(reference)), THRESHOLDS["gradient_norm_floor"])
        ),
        "reference_norm_per_sample": norm,
        "all_reference_rows_nonzero": bool(np.all(norm > THRESHOLDS["gradient_norm_floor"])),
        "finite": finite,
        "passed": bool(
            finite
            and np.all(norm > THRESHOLDS["gradient_norm_floor"])
            and np.all(relative <= THRESHOLDS["gradient_relative"])
        ),
    }


def compare(a, b, case, constants):
    residuals = {key: physical(value, case, constants) for key, value in (("A", a), ("B", b))}
    cost = gradient_error(b["cost_gradient"], a["cost_gradient"])
    state = gradient_error(b["state_gradient"], a["state_gradient"])
    u0 = np.max(np.abs(a["u0"] - b["u0"]), axis=-1)
    trajectory = max(
        float(np.max(np.abs(a["states"] - b["states"]))), float(np.max(np.abs(a["controls"] - b["controls"])))
    )
    finite = all(np.isfinite(value).all() for snapshot in (a, b) for value in snapshot.values())
    solved = all(
        np.all(snapshot["status"] == 0)
        and np.all(snapshot["valid"])
        and np.all(snapshot["convergence"] <= SOLVER["tol_convergence"])
        for snapshot in (a, b)
    )
    gates = {
        "finite": bool(finite),
        "all_rows_solved": bool(solved),
        "u0": bool(np.isfinite(u0).all() and np.max(u0) <= THRESHOLDS["u0_max_abs"]),
        "cost_gradient": cost["passed"],
        "state_gradient": state["passed"],
        "physical": all(v["finite"] and v["maximum"] <= THRESHOLDS["physical_violation"] for v in residuals.values()),
        "stable_interior_point": all(
            np.all(v["interior_margin_per_sample"] > THRESHOLDS["stable_interior_margin"]) for v in residuals.values()
        ),
    }
    return {
        "u0_max_abs": float(np.max(u0)),
        "u0_per_sample_max_abs": u0,
        "full_trajectory_max_abs": trajectory,
        "full_controls_max_abs": float(np.max(np.abs(a["controls"] - b["controls"]))),
        "cost_gradient": cost,
        "state_gradient": state,
        "physical": residuals,
        "gates": gates,
        "passed": all(gates.values()),
    }


def finite_difference(runtime, analytic, case, constants, artifacts, prefix):
    """Coordinate differences of every row's own loss, with two fixed steps.

    Both models are block independent. Perturbing the same coordinate in every
    row simultaneously computes every row derivative without summing losses.
    No unstable or failed perturbation is discarded.
    """
    import torch

    results = []
    for step in (1e-3, 5e-4):
        gradients = {}
        solved, finite, stable = True, True, True
        worst_residual = 0.0
        for variable in ("costs", "state"):
            numerical = np.empty_like(case[variable])
            for column in range(case[variable].shape[1]):
                losses = []
                for sign in (1.0, -1.0):
                    inputs = {key: case[key].copy() for key in ("costs", "state")}
                    inputs[variable][:, column] += sign * step
                    costs = torch.as_tensor(inputs["costs"], device="cuda:0", dtype=torch.float64)
                    state = torch.as_tensor(inputs["state"], device="cuda:0", dtype=torch.float64)
                    mean, solution, callbacks = runtime.forward(costs, state)
                    losses.append((0.5 * mean.square().sum(-1)).detach().cpu().numpy())
                    snap = runtime.snapshot((mean, solution, None, callbacks))
                    residual = physical(snap, {**case, "state": inputs["state"]}, constants)
                    solved &= bool(
                        np.all(snap["status"] == 0)
                        and np.all(snap["valid"])
                        and np.all(snap["convergence"] <= SOLVER["tol_convergence"])
                    )
                    finite &= bool(residual["finite"] and np.isfinite(losses[-1]).all())
                    stable &= bool(
                        np.all(residual["interior_margin_per_sample"] > THRESHOLDS["stable_interior_margin"])
                    )
                    worst_residual = max(worst_residual, residual["maximum"])
                numerical[:, column] = (losses[0] - losses[1]) / (2 * step)
            key = "cost_gradient" if variable == "costs" else "state_gradient"
            gradients[key] = numerical
            artifacts[f"{prefix}_fd_{step}_{key}"] = numerical
        checks = {key: gradient_error(analytic[key], value) for key, value in gradients.items()}
        results.append(
            {
                "step": step,
                "gradients": checks,
                "all_perturbations_solved": solved,
                "finite": finite,
                "all_perturbations_interior": stable,
                "physical_maximum": worst_residual,
                "passed": bool(
                    solved
                    and finite
                    and stable
                    and worst_residual <= THRESHOLDS["physical_violation"]
                    and all(value["passed"] for value in checks.values())
                ),
            }
        )
    return {
        "method": "all cost/state coordinates, central differences, per-row squared-u0 loss",
        "steps": results,
        "passed": all(value["passed"] for value in results),
    }


def timing_summary(pairs):
    a = np.asarray([pair["A_ms"] for pair in pairs])
    b = np.asarray([pair["B_ms"] for pair in pairs])
    p50 = float(np.median(b) / np.median(a))
    p95 = float(np.percentile(b, 95) / np.percentile(a, 95))
    # Circular paired block bootstrap preserves A/B pairing and short temporal
    # correlation; its interval is reported, not substituted for the specified gate.
    rng = np.random.default_rng(318)
    length = min(10, len(pairs))
    indexes = (rng.integers(len(pairs), size=(1000, (len(pairs) + length - 1) // length, 1)) + np.arange(length)) % len(
        pairs
    )
    indexes = indexes.reshape(1000, -1)[:, : len(pairs)]
    boot50 = np.median(b[indexes], axis=1) / np.median(a[indexes], axis=1)
    boot95 = np.percentile(b[indexes], 95, axis=1) / np.percentile(a[indexes], 95, axis=1)
    return {
        "A_p50_ms": float(np.median(a)),
        "B_p50_ms": float(np.median(b)),
        "A_p95_ms": float(np.percentile(a, 95)),
        "B_p95_ms": float(np.percentile(b, 95)),
        "p50_ratio_B_over_A": p50,
        "p95_ratio_B_over_A": p95,
        "paired_block_bootstrap_95pct": {
            "block_pairs": length,
            "p50_ratio": np.percentile(boot50, [2.5, 97.5]),
            "p95_ratio": np.percentile(boot95, [2.5, 97.5]),
        },
        "passed": bool(
            np.isfinite(a).all()
            and np.isfinite(b).all()
            and np.all(a > 0)
            and np.all(b > 0)
            and p50 <= THRESHOLDS["latency_ratio"]
            and p95 <= THRESHOLDS["latency_ratio"]
        ),
    }


def existing_diagnostics(directory):
    result = {"directory": str(directory), "case": "previous 9-state jerk extension diagnostic", "comparisons": []}
    for batch in REQUIRED_BATCHES:
        a_path, b_path = [directory / f"{backend}-b{batch}.json" for backend in ("bridge", "torch")]
        if not all(path.is_file() and path.with_suffix(".npz").is_file() for path in (a_path, b_path)):
            continue
        try:
            a_json, b_json = json.loads(a_path.read_text()), json.loads(b_path.read_text())
            with (
                np.load(a_path.with_suffix(".npz"), allow_pickle=False) as a,
                np.load(b_path.with_suffix(".npz"), allow_pickle=False) as b,
            ):
                comparison = gradient_error(b["cost_gradient"], a["cost_gradient"])
                u0_error = float(np.max(np.abs(a["trajectory"][:, 0, 9:] - b["trajectory"][:, 0, 9:])))
                gates = {
                    "cost_gradient": comparison["passed"],
                    "u0": bool(np.isfinite(u0_error) and u0_error <= THRESHOLDS["u0_max_abs"]),
                    "solved": bool(np.all(a["status"] == 0) and np.all(b["status"] == 0)),
                    "finite_difference": all(
                        np.isfinite(value.get("finite_difference", {}).get("relative_error", np.inf))
                        and value.get("finite_difference", {}).get("relative_error", np.inf)
                        <= THRESHOLDS["gradient_relative"]
                        for value in (a_json, b_json)
                    ),
                }
                result["comparisons"].append(
                    {
                        "batch": batch,
                        "cost_gradient": comparison,
                        "u0_max_abs": u0_error,
                        "numerical_gates": gates,
                        "passed": all(gates.values()),
                        "A_finite_difference": a_json.get("finite_difference"),
                        "B_finite_difference": b_json.get("finite_difference"),
                        "files": {
                            str(path): sha256(path)
                            for path in (a_path, b_path, a_path.with_suffix(".npz"), b_path.with_suffix(".npz"))
                        },
                    }
                )
        except Exception as error:
            result.setdefault("read_errors", []).append(f"{batch}: {type(error).__name__}: {error}")
    result["completed_batches"] = [value["batch"] for value in result["comparisons"]]
    result["incomplete"] = result["completed_batches"] != list(REQUIRED_BATCHES)
    result["unresolved"] = bool(
        result["incomplete"] or result.get("read_errors") or any(not value["passed"] for value in result["comparisons"])
    )
    result["qualification"] = (
        "Preserved legacy custom-backend history only; missing or failed historical files do not gate M4. "
        "The explicit official-regression report determines the current fixed-jerk regression result."
    )
    result["blocks_m4"] = False
    return result


OFFICIAL_REGRESSION_CHECKS = (
    "base_solved_finite", "raw_vjp_and_callbacks_finite", "full_controls_vs_native",
    "cost_gradient_vs_native", "state_gradient_vs_native", "base_and_perturbations_physically_feasible",
    "perturbations_solved_finite", "stable_active_masks", "official_full_gradients_vs_both_fd_steps",
    "official_old_cost_direction_vs_both_fd_steps", "installed_baseline_unchanged",
)


def _upstream_source_index(hashes):
    """Compare pinned source content independently of the checkout's location."""
    result = {}
    for name, digest in hashes.items():
        name = Path(name).as_posix()
        if "/examples/rl/util/" in name:
            key = "examples/rl/util/" + name.rsplit("/examples/rl/util/", 1)[1]
        elif "/turbompc/" in name:
            key = "turbompc/" + name.rsplit("/turbompc/", 1)[1]
        else:
            raise ValueError(f"Unrecognized upstream source record: {name}")
        if key in result and result[key] != digest:
            raise ValueError(f"Conflicting source hashes: {key}")
        result[key] = digest
    return result


def validate_official_regression(path, measured_library_hashes, measured_source_hashes):
    """Fail closed on a different A, different fixed input, or failed official report.

    The companion NPZ is self-contained: no legacy custom-backend files are
    needed. Inspect saved numerical arrays as well as the producer's booleans.
    This validates recorded evidence; it does not rerun GPU calculations.
    """
    path = Path(path).resolve()
    result = {
        "path": str(path), "passed": False, "checks": {},
        "scope": "Fixed B1/H15 jerk regression through official MPCBackend + MPC.forward, unchanged original A only",
    }
    checks = result["checks"]
    try:
        data = path.read_bytes()
        document = json.loads(data)
        result["report_sha256"] = hashlib.sha256(data).hexdigest()
        kkt = {Path(name).resolve(): digest for name, digest in measured_library_hashes.items()
               if Path(name).name == "libcudss_sparse_kkt_ffi.so"}
        if len(kkt) != 1:
            raise ValueError("Exactly one measured original-A KKT library is required")
        expected_path, expected_hash = next(iter(kkt.items()))
        if re.fullmatch(r"[0-9a-f]{64}", expected_hash) is None:
            raise ValueError("Invalid measured original-A KKT hash")
        provenance = document["provenance"]
        checks["original_role"] = (
            provenance.get("kkt_role") == "original"
            and document.get("role") == "official full-plan jerk accuracy diagnostic: original"
            and provenance.get("upstream_commit") == REVISION
        )
        checks["same_measured_A_library"] = all(
            provenance.get(key) == expected_hash for key in (
                "expected_kkt_library_sha256", "actual_loaded_kkt_sha256",
                "installed_baseline_sha256_before", "installed_baseline_sha256_after",
            )
        ) and all(
            Path(provenance[key]).resolve() == expected_path for key in (
                "expected_kkt_library", "actual_loaded_kkt_library", "installed_baseline_library",
            )
        )
        result["measured_A_kkt"] = {"path": str(expected_path), "sha256": expected_hash}
        expected_sources = _upstream_source_index(measured_source_hashes)
        actual_sources = _upstream_source_index(provenance["verified_upstream_files"])
        required_sources = {"examples/rl/util/mpc.py", "examples/rl/util/jax_torch.py"}
        checks["same_pinned_official_sources"] = (
            required_sources <= expected_sources.keys() and actual_sources == expected_sources
        )
        implementation = document["implementation"]
        checks["official_full_plan_interface"] = (
            implementation.get("backend_class") == "_jerk_official_backend.MPCBackend"
            and implementation.get("bridge_class") == "_jerk_official_bridge.MPC"
            and implementation.get("entry_point") == "official MPC.forward (mean, full controls)"
            and all(provenance["verified_upstream_files"].get(implementation.get(key)) == expected_sources[suffix]
                    for key, suffix in (("backend_file", "examples/rl/util/mpc.py"),
                                        ("bridge_file", "examples/rl/util/jax_torch.py")))
        )
        expected_case = {"batch": 1, "horizon": 15, "nx": 9, "nu": 3, "dtype": "float64", "seed": 719,
                         "loss": "sum(all 16 control nodes * original saved probe)"}
        checks["fixed_case_metadata"] = all(document.get("case", {}).get(k) == v for k, v in expected_case.items())
        settings = implementation["effective_solver_parameters"]
        expected_settings = {"constraint_handling": "projection", "log_barrier_kappa": 0.0,
                             "tol_convergence": 1e-6, "convergence_criterion": "first_order",
                             "forward_hessian_min_eig": 1e-12, "num_sqp_iteration_max": 15}
        expected_admm = ADMM | {"eps_abs": 1e-8, "eps_rel": 1e-8}
        checks["fixed_solver_settings"] = (
            all(settings.get(key) == value for key, value in expected_settings.items())
            and all(float(settings["admm"][key]) == float(value) for key, value in expected_admm.items())
            and settings["linesearch"]["method"] == "merit"
            and settings["linesearch"]["alphas"] == SOLVER["linesearch"]["alphas"]
            and all(float(settings["linesearch"]["merit"][key]) == float(value)
                    for key, value in SOLVER["linesearch"]["merit"].items())
        )
        checks["reported_checks_passed"] = (
            "error" not in document and document.get("diagnostic_checks_passed") is True
            and all(document.get("checks", {}).get(key) is True for key in OFFICIAL_REGRESSION_CHECKS)
        )
        helper = load_example("_acceptance_fixed_jerk_inputs", ROOT / "scripts/turbompc_benchmark_common.py")
        case = helper.make_case(1, 15, False)
        arrays_path = path.with_suffix(".npz")
        raw_arrays = arrays_path.read_bytes()
        result.update(arrays_file=str(arrays_path), arrays_sha256=hashlib.sha256(raw_arrays).hexdigest(),
                      expected_inputs_sha256=array_hash({k: np.asarray(v) for k, v in case.items()}))
        with np.load(io.BytesIO(raw_arrays), allow_pickle=False) as saved:
            checks["exact_fixed_inputs"] = provenance.get("same_saved_inputs") is True and all(
                f"input_{key}" in saved and saved[f"input_{key}"].dtype == np.asarray(value).dtype
                and np.array_equal(saved[f"input_{key}"], value) for key, value in case.items()
            )
            shapes = {"base_trajectory": (1, 16, 12), "native_trajectory": (1, 16, 12),
                      "base_returned_controls": (1, 16, 3), "callback_mpc_cost_adjoint": (1, 12),
                      "callback_mpc_context_adjoint": (1, 9)}
            for name, shape in (("cost_gradient", (1, 12)), ("state_gradient", (1, 9))):
                for prefix in ("base", "native", "raw_retained", "fd_h0", "fd_h1"):
                    shapes[f"{prefix}_{name}"] = shape
            checks["raw_arrays_finite_and_complete"] = all(
                key in saved and saved[key].shape == shape and saved[key].dtype == np.float64
                and np.isfinite(saved[key]).all() for key, shape in shapes.items()
            )
            if not checks["raw_arrays_finite_and_complete"]:
                raise ValueError("Official regression arrays are missing, nonfinite, or incorrectly shaped")
            checks["raw_base_solved"] = (
                saved["base_status"].shape == (1,) and np.all(saved["base_status"] == 0)
                and saved["base_valid"].shape == (1,) and np.all(saved["base_valid"])
            )
            checks["controls_and_physical_constraints"] = bool(
                np.max(np.abs(saved["base_returned_controls"] - saved["native_trajectory"][..., 9:])) <= THRESHOLDS["u0_max_abs"]
                and np.max(np.abs(saved["base_returned_controls"] - saved["base_trajectory"][..., 9:])) <= 1e-8
                and max(helper.physical_residuals(saved["base_trajectory"], case, False).values()) <= THRESHOLDS["physical_violation"]
            )
            result["recomputed_gradient_comparisons"] = {}
            for name in ("cost_gradient", "state_gradient"):
                comparisons = {"official_vs_native": gradient_error(saved[f"base_{name}"], saved[f"native_{name}"])}
                for index in range(2):
                    comparisons[f"official_vs_fd_h{index}"] = gradient_error(saved[f"base_{name}"], saved[f"fd_h{index}_{name}"])
                result["recomputed_gradient_comparisons"][name] = comparisons
                checks[f"raw_{name}_matches_reference_and_fd"] = all(value["passed"] for value in comparisons.values())
        fd = document["finite_difference"]
        steps, perturbations = fd["steps"], fd["perturbations"]
        checks["complete_stable_fd_evidence"] = (
            [step["step"] for step in steps] == [1e-4, 5e-5]
            and fd.get("forward_perturbations") == len(perturbations) == 88
            and all(fd.get(key) is True for key in ("all_solved", "all_finite", "all_active_masks_equal", "all_mask_replay_plans_match"))
            and all(all(row.get(key) is True for key in ("solved", "finite", "active_masks_equal", "mask_replay_solved", "mask_duals_finite"))
                    for row in perturbations)
            and all(np.isfinite(row["physical_maximum"])
                    and row["physical_maximum"] <= THRESHOLDS["physical_violation"] for row in perturbations)
        )
        checks["old_direction_matches_both_fd_steps"] = all(
            np.isfinite(step["old_direction"]["official_relative_error"])
            and step["old_direction"]["official_relative_error"] <= THRESHOLDS["gradient_relative"] for step in steps
        )
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    result["checks"] = {key: bool(value) for key, value in checks.items()}
    result["passed"] = "error" not in result and bool(checks) and all(result["checks"].values())
    return result


def apply_official_regression_gate(report, regression):
    """Historical custom-backend outcomes never override current official evidence."""
    report["official_regression"] = regression
    report["prior_failure_blocks_m4"] = False
    report["official_regression_blocks_m4"] = regression.get("passed") is not True
    report["m4_gate_passed"] = report.get("selected_linear_suite_passed") is True and regression.get("passed") is True


def reevaluate_gate_report(measurement_path, official_path, output, measurement_commit="a771284"):
    """Write a separate CPU-only decision; never relabel or overwrite GPU evidence."""
    measurement_path, output = Path(measurement_path).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Choose a new gate re-evaluation report; original evidence is preserved")
    raw = measurement_path.read_bytes()
    measured = json.loads(raw)
    revision = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", measurement_commit], text=True).strip()
    source = subprocess.check_output(["git", "-C", str(ROOT), "show", f"{revision}:scripts/turbompc/check_acceptance.py"])
    if hashlib.sha256(source).hexdigest() != measured["harness_sha256"] or measured["upstream_revision"] != REVISION:
        raise ValueError("Saved GPU measurements do not match the declared unchanged-A harness checkpoint")
    regression = validate_official_regression(official_path, measured["loaded_solver_hashes"], measured["verified_upstream_sources"])
    arrays = Path(measured["arrays_file"])
    report = {
        "scope": "CPU-only M4 gate re-evaluation for unchanged original A; no authorization to replace A",
        "gpu_measurements_rerun": False, "time_utc": datetime.now(UTC).isoformat(),
        "measurement_report": {"path": str(measurement_path), "sha256": hashlib.sha256(raw).hexdigest(),
                               "arrays_file": str(arrays), "arrays_sha256": sha256(arrays),
                               "harness_commit": revision, "harness_sha256": measured["harness_sha256"]},
        "gate_harness_sha256": sha256(__file__),
        "measured_linear_gates": measured["gates"], "measured_m4_gate_passed": measured["m4_gate_passed"],
        "selected_linear_suite_passed": measured["selected_linear_suite_passed"] is True and bool(measured["gates"])
                                        and all(value is True for value in measured["gates"].values()),
        "historical_custom_backend_scope": "Retained in the original measurement report as diagnostic history only",
    }
    apply_official_regression_gate(report, regression)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(json_safe(report), handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(json_safe(report), indent=2, allow_nan=False))
    return 0 if report["m4_gate_passed"] else 1


def loaded_libraries():
    return sorted(
        {
            line.split()[-1]
            for line in Path("/proc/self/maps").read_text().splitlines()
            if len(line.split()) >= 6
            and any(
                name in line
                for name in (
                    "_turbompc_native",
                    "_ffi.so",
                    "libcudss",
                    "libcudart",
                    "libcublas",
                    "libcusparse",
                    "libnvrtc",
                    "libnvJitLink",
                )
            )
        }
    )


def run_batch(args, batch, report, artifacts):
    import torch

    case = case_bank(batch)
    entry = {
        "batch": batch,
        "case_sha256": array_hash(case),
        "cold": {},
        "pairs": [],
        "input_generation": "seed 719; fixed 256-row bank prefixes; no row filtering",
    }
    report["batches"].append(entry)
    for key, value in case.items():
        artifacts[f"b{batch}_input_{key}"] = value
    runtimes, cold = {}, {}
    try:
        for label in ("A", "B"):
            start = time.perf_counter()
            runtime = (
                Baseline(case, args.reference) if label == "A" else Native(case, runtimes["A"].constants, args.compile)
            )
            runtimes[label] = runtime
            setup = time.perf_counter() - start
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = full(runtime, capture=label == "A")
            torch.cuda.synchronize()
            entry["cold"][label] = {
                "setup_seconds": setup,
                "first_full_seconds_including_jit": time.perf_counter() - start,
            }
            cold[label] = runtime.snapshot(result)
            for key, value in cold[label].items():
                artifacts[f"b{batch}_{label}_{key}"] = value
            if label == "A":
                raw_report, raw_arrays = runtime.raw_gradient_check(result)
                entry["A_raw_gradient_before_timing"] = raw_report
                for key, value in raw_arrays.items():
                    artifacts[f"b{batch}_A_raw_{key}"] = value
            del result
        constants = runtimes["A"].constants
        entry["effective_A_solver"] = runtimes["A"].backend.solver.params
        entry["effective_B_sqp"] = asdict(runtimes["B"].solver.config)
        entry["effective_B_admm"] = asdict(runtimes["B"].solver.admm.config)
        a_config, b_config = entry["effective_A_solver"], entry["effective_B_admm"]
        entry["configuration_checks"] = {
            "forward_admm_parameters": all(
                a_config["admm"][key] == value and b_config["alpha" if key == "relaxation_parameter" else key] == value
                for key, value in ADMM.items()
            ),
            "sqp_iterations": a_config["num_sqp_iteration_max"] == entry["effective_B_sqp"]["max_iter"] == 15,
            "sqp_tolerance": a_config["tol_convergence"]
            == entry["effective_B_sqp"]["tolerance"]
            == SOLVER["tol_convergence"],
            "first_order_projection_without_slack": a_config["convergence_criterion"] == "first_order"
            and a_config["constraint_handling"] == "projection"
            and b_config["constraint_handling"] == 0
            and a_config["log_barrier_kappa"] == b_config["log_barrier_kappa"] == 0.0
            and not runtimes["A"].backend.solver.program.use_slack_variables
            and not b_config["use_slack"],
            "line_search": a_config["linesearch"]["method"] == entry["effective_B_sqp"]["linesearch"] == "merit"
            and tuple(a_config["linesearch"]["alphas"]) == tuple(entry["effective_B_sqp"]["alphas"])
            and a_config["linesearch"]["merit"]["armijo_coefficient"] == entry["effective_B_sqp"]["armijo"],
            "official_forward_backend": runtimes["A"].backend.solver.forward_backend.name.lower()
            == SOLVER["forward_backend"],
            "official_backward_backend": runtimes["A"].backend.solver.backward_backend.name.lower()
            == SOLVER["backward_backend"],
        }
        entry["correctness"] = compare(cold["A"], cold["B"], case, constants)
        entry["states"] = {
            label: {key: value[key] for key in ("status", "valid", "convergence", "iterations", "admm_iterations")}
            for label, value in cold.items()
        }
        entry["warmup_full_ms"] = {"A": [], "B": []}
        for index in range(args.warmup):
            for label in ("A", "B") if index % 2 == 0 else ("B", "A"):
                torch.cuda.synchronize()
                start = time.perf_counter()
                result = full(runtimes[label])
                torch.cuda.synchronize()
                entry["warmup_full_ms"][label].append(1000 * (time.perf_counter() - start))
                del result
        for index in range(args.pairs):
            pair = {
                "index": index,
                "order": ["A", "B"] if index % 2 == 0 else ["B", "A"],
                "start_time_ns": time.time_ns(),
            }
            samples = {}
            for label in pair["order"]:
                torch.cuda.synchronize()
                start = time.perf_counter()
                result = full(runtimes[label])
                torch.cuda.synchronize()
                pair[f"{label}_ms"] = 1000 * (time.perf_counter() - start)
                # Transfers/validation are outside the measured full forward +
                # backward. Retain every pair, including failed and slow pairs.
                samples[label] = runtimes[label].snapshot(result)
                del result
            pair["end_time_ns"] = time.time_ns()
            comparison = compare(samples["A"], samples["B"], case, constants)
            pair["numerical_passed"] = comparison["passed"]
            pair["u0_max_abs"] = comparison["u0_max_abs"]
            pair["cost_gradient_max_relative"] = comparison["cost_gradient"]["maximum_relative"]
            pair["state_gradient_max_relative"] = comparison["state_gradient"]["maximum_relative"]
            pair["physical_maximum"] = max(v["maximum"] for v in comparison["physical"].values())
            pair["all_solved"] = comparison["gates"]["all_rows_solved"]
            entry["pairs"].append(pair)
            if index == 0 or (index + 1) % 10 == 0 or index + 1 == args.pairs:
                print(f"batch {batch}: paired full F+B {index + 1}/{args.pairs}", flush=True)
        entry["timing"] = timing_summary(entry["pairs"])
        result = full(runtimes["A"], capture=True)
        raw_report, raw_arrays = runtimes["A"].raw_gradient_check(result)
        entry["A_raw_gradient_after_timing"] = raw_report
        for key, value in raw_arrays.items():
            artifacts[f"b{batch}_A_raw_after_{key}"] = value
        del result
        entry["finite_difference"] = {
            label: finite_difference(runtime, cold[label], case, constants, artifacts, f"b{batch}_{label}")
            for label, runtime in runtimes.items()
        }
        entry["gates"] = {
            "same_shared_solver_configuration": all(entry["configuration_checks"].values()),
            "correctness": entry["correctness"]["passed"],
            "raw_vjp_and_callbacks": entry["A_raw_gradient_before_timing"]["raw_vjp_and_callbacks_finite"]
            and entry["A_raw_gradient_after_timing"]["raw_vjp_and_callbacks_finite"],
            "every_timed_pair_numerically_valid": all(pair["numerical_passed"] for pair in entry["pairs"]),
            "finite_difference": all(value["passed"] for value in entry["finite_difference"].values()),
            "at_least_100_pairs": len(entry["pairs"]) >= 100,
            "performance": entry["timing"]["passed"],
        }
        entry["passed"] = all(entry["gates"].values())
    except Exception as error:
        entry.update(passed=False, error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        print(f"batch {batch} failed: {entry['error']}", flush=True)
    finally:
        for runtime in reversed(list(runtimes.values())):
            try:
                runtime.close()
            except Exception as error:
                entry.setdefault("close_errors", []).append(f"{type(error).__name__}: {error}")
                entry["passed"] = False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batches", type=int, nargs="+", default=list(REQUIRED_BATCHES))
    parser.add_argument("--pairs", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--reference", type=Path, default=Path(os.environ.get("TURBOMPC_BASELINE_SOURCE", ROOT / "build/reference"))
    )
    parser.add_argument("--upstream-repo", type=Path, default=ROOT / "build/reference")
    parser.add_argument(
        "--prior-diagnostic-dir", type=Path, default=ROOT / "results/turbompc-port/sweep-linear-cu128-v1"
    )
    parser.add_argument(
        "--official-regression-report", type=Path,
        default=ROOT / "results/turbompc-port/official-jerk-original-b1-v1.json",
        help="Required fixed-B1 official jerk evidence for the same unchanged original A",
    )
    parser.add_argument("--reevaluate-report", type=Path, help="CPU-only: re-evaluate the gate of saved GPU measurements")
    parser.add_argument("--measurement-harness-commit", default="a771284", help="Checkpoint that actually collected saved GPU measurements")
    args = parser.parse_args()
    if args.reevaluate_report is not None:
        return reevaluate_gate_report(args.reevaluate_report, args.official_regression_report, args.output,
                                      args.measurement_harness_commit)
    if min(args.pairs, args.warmup, *args.batches) < 1 or len(set(args.batches)) != len(args.batches):
        parser.error("positive distinct batches, positive pairs and warmup are required")
    args.output = args.output.resolve()
    args.reference = args.reference.resolve()
    npz_path = args.output.with_suffix(".npz")
    if args.output.exists() or npz_path.exists():
        raise FileExistsError("Use a fresh output name; previous results are preserved")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "scope": "M4 official linear OCP adaptation; A official MPCBackend + MPC.mean_only, B TorchMPC",
        "time_utc": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "executable": sys.executable,
        "harness_sha256": sha256(__file__),
        "upstream_revision": REVISION,
        "horizon": N,
        "dtype": "float64",
        "dt": DT,
        "nx": NX,
        "nu": NU,
        "thresholds": THRESHOLDS,
        "requested_batches": args.batches,
        "requested_pairs": args.pairs,
        "solver_contract": SOLVER,
        "loss": "sum over rows of 0.5 * ||u0||^2; no batch averaging",
        "problem_overrides": {
            "source": "official linear.yaml, Euler dynamics",
            "stage_weight": STAGE_WEIGHT,
            "final_weight": FINAL_WEIGHT,
            "control_weight": CONTROL_WEIGHT,
            "state_bounds": [-5.0, 5.0],
            "control_bounds": [-2.0, 2.0],
            "reference": "zero",
            "seed": "repeat initial state; zero controls; valid explicit primal seed",
            "cost_parameters": "10 log multipliers: stage state 4, final state 4, control 2",
        },
        "timing_contract": {
            "operation": "Torch input costs/state -> u0 -> squared loss -> both gradients -> device completion",
            "schedule": "serial AB, BA alternating pairs; no outlier rejection or retries",
            "cold": "backend setup and first full call (includes JIT/cache lookup) reported separately from warm pairs",
            "validation": "outside timed intervals; both runtimes stay resident in this one process",
            "shared_gpu": True,
            "confidence_intervals": "diagnostic; gate uses measured P50/P95 ratios",
        },
        "environment": {
            key: os.environ.get(key)
            for key in (
                "TURBOMPC_CUDA_PROFILE",
                "CUDA_HOME",
                "CUDA_VISIBLE_DEVICES",
                "LD_LIBRARY_PATH",
                "TORCHINDUCTOR_CACHE_DIR",
                "TRITON_CACHE_DIR",
            )
        },
        "arrays_file": str(npz_path),
        "batches": [],
        "m4_gate_passed": False,
        "gpu_load_columns": {
            "gpus": "uuid,name,driver_version,memory.used MiB,memory.total MiB,utilization.gpu %,utilization.memory %,power.draw W,temperature.gpu C,clocks.sm MHz,pstate",
            "compute_processes": "gpu_uuid,pid,process_name,used_gpu_memory MiB",
        },
    }
    artifacts, monitor = {}, GPULoad()
    monitor.start()
    try:
        start = time.perf_counter()
        if os.environ.get("TURBOMPC_CUDA_PROFILE") != "cu128":
            raise RuntimeError("Run through bash scripts/turbompc/env.sh cu128")
        os.environ.setdefault("MAX_JOBS", "2")
        os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "2")
        os.environ.setdefault("TURBOMPC_NATIVE_BUILD_DIR", str(ROOT / "build/turbompc/native/cu128"))
        os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(ROOT / "build/inductor-cu128"))
        os.environ.setdefault("TRITON_CACHE_DIR", str(ROOT / "build/triton-cu128"))
        report["environment"].update(
            {
                key: os.environ.get(key)
                for key in (
                    "TURBOMPC_NATIVE_BUILD_DIR",
                    "TORCHINDUCTOR_CACHE_DIR",
                    "TRITON_CACHE_DIR",
                    "MAX_JOBS",
                    "TORCHINDUCTOR_COMPILE_THREADS",
                )
            }
        )
        sys.path.insert(0, str(ROOT / "src"))
        import jax
        import torch
        import turbompc

        jax.config.update("jax_enable_x64", True)
        if (
            jax.__version__ != "0.6.2"
            or not torch.__version__.startswith("2.11.0")
            or torch.version.cuda != "12.8"
            or not torch.cuda.is_available()
        ):
            raise RuntimeError("Requires JAX 0.6.2 and PyTorch 2.11.0 + CUDA 12.8 on a GPU")
        if any(device.platform not in {"gpu", "cuda"} for device in jax.devices()):
            raise RuntimeError("A must use GPU JAX")
        installed = Path(turbompc.__file__).resolve().parent
        if not installed.is_relative_to(Path(sys.prefix).resolve()):
            raise RuntimeError(f"A package must come from the selected cu128 venv: {installed}")
        report["verified_upstream_sources"] = verify_original(args.reference, args.upstream_repo, installed)
        report["build_records_before"] = verified_build_records()
        report["versions"] = {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "jax",
                "jaxlib",
                "turbompc",
                "nvidia-cudss-cu12",
            )
        }
        report["gpu"] = {
            "torch": torch.cuda.get_device_name(0),
            "torch_uuid": str(getattr(torch.cuda.get_device_properties(0), "uuid", "unavailable")),
            "jax": [str(device) for device in jax.devices()],
        }
        report["nvcc"] = subprocess.check_output(
            [str(Path(os.environ["CUDA_HOME"]) / "bin/nvcc"), "--version"],
            text=True,
        ).strip()
        native_source = ROOT / "native/turbompc/upstream/backward/cudss_sparse_kkt.cu"
        match = re.search(r"int\s+refinement_steps\s*=\s*(\d+)", native_source.read_text())
        report["backward_ir_difference"] = {
            "A": {
                "setting": "unmodified pinned kernel leaves cuDSS IR at library default",
                "expected_default_steps": 0,
            },
            "B": {
                "source_refinement_steps": int(match.group(1)) if match else None,
                "source": str(native_source),
                "source_sha256": sha256(native_source),
            },
            "qualification": "A IR0 expectation vs B IR2 is an explicit native correction; source inspection is not a runtime config readback.",
        }
        report["native_python_sources"] = {
            str(path): sha256(path) for path in (ROOT / "src/isaaclab_ipc/turbompc").glob("*.py")
        }
        report["shared_startup_seconds"] = time.perf_counter() - start
        for batch in args.batches:
            run_batch(args, batch, report, artifacts)
        report["loaded_libraries"] = loaded_libraries()
        solver_libraries = [
            Path(path)
            for path in report["loaded_libraries"]
            if "_turbompc_native" in path or Path(path).name.endswith("_ffi.so")
        ]
        report["loaded_solver_hashes"] = {str(path): sha256(path) for path in solver_libraries}
        report["build_records_after"] = verified_build_records()
        expected_artifacts = {
            str(Path(path).resolve()): digest
            for record in report["build_records_before"].values()
            for path, digest in record["artifact_sha256"].items()
        }
        report["build_records_match_loaded_artifacts"] = (
            report["build_records_before"] == report["build_records_after"]
            and bool(report["loaded_solver_hashes"])
            and all(
                expected_artifacts.get(str(Path(path).resolve())) == digest
                for path, digest in report["loaded_solver_hashes"].items()
            )
        )
        cuda_libraries = [
            Path(path)
            for path in report["loaded_libraries"]
            if Path(path).name.startswith(
                ("libcudss", "libcudart", "libcublas", "libcusparse", "libnvrtc", "libnvJitLink")
            )
        ]
        required = {"libadmm_cudss_ffi.so", "libcudss_sparse_kkt_ffi.so"}
        ffi = [path for path in solver_libraries if path.name in required]
        native = [path for path in solver_libraries if "_turbompc_native" in path.name]
        report["loaded_solver_provenance_passed"] = (
            {path.name for path in ffi} == required
            and all(path.resolve().is_relative_to(Path(sys.prefix).resolve()) for path in ffi)
            and len(native) == 1
            and native[0].resolve().is_relative_to((ROOT / "build/turbompc/native/cu128").resolve())
            and all(
                path.resolve().is_relative_to(Path(sys.prefix).resolve())
                or path.resolve().is_relative_to(Path(os.environ["CUDA_HOME"]).resolve())
                for path in cuda_libraries
            )
            and not any("libcudart.so.13" in path for path in report["loaded_libraries"])
        )
    except Exception as error:
        report.update(error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
    finally:
        monitor.close()
        report["gpu_load_samples"] = monitor.samples
        report["native_python_sources_after"] = {
            path: sha256(Path(path)) if Path(path).is_file() else None
            for path in report.get("native_python_sources", {})
        }
        report["native_sources_unchanged"] = bool(report.get("native_python_sources")) and (
            report["native_python_sources_after"] == report["native_python_sources"]
        )
        report["harness_source_unchanged"] = (
            Path(__file__).is_file() and sha256(Path(__file__)) == report["harness_sha256"]
        )
        report["prior_extension_diagnostic"] = existing_diagnostics(args.prior_diagnostic_dir)
        report["gates"] = {
            "all_required_batches": sorted(args.batches) == list(REQUIRED_BATCHES),
            "all_requested_batches_passed": len(report["batches"]) == len(args.batches)
            and all(value.get("passed", False) for value in report["batches"]),
            "loaded_library_paths": bool(report.get("loaded_solver_provenance_passed", False)),
            "build_records_match_loaded_artifacts": bool(report.get("build_records_match_loaded_artifacts", False)),
            "source_files_unchanged_during_run": report["native_sources_unchanged"]
            and report["harness_source_unchanged"],
            "gpu_load_recorded": any(sample.get("gpus", {}).get("returncode") == 0 for sample in monitor.samples),
            "no_harness_error": "error" not in report,
        }
        report["selected_linear_suite_passed"] = all(report["gates"].values())
        regression = validate_official_regression(
            args.official_regression_report, report.get("loaded_solver_hashes", {}),
            report.get("verified_upstream_sources", {}),
        )
        apply_official_regression_gate(report, regression)
        report["completed_utc"] = datetime.now(UTC).isoformat()
        with npz_path.open("xb") as arrays_handle:
            np.savez_compressed(arrays_handle, **artifacts)
        with args.output.open("x") as handle:
            json.dump(json_safe(report), handle, indent=2, allow_nan=False)
            handle.write("\n")
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "selected_linear_suite_passed": report["selected_linear_suite_passed"],
                    "prior_failure_blocks_m4": report["prior_failure_blocks_m4"],
                    "m4_gate_passed": report["m4_gate_passed"],
                    "error": report.get("error"),
                },
                indent=2,
            ),
            flush=True,
        )
    return 0 if report["m4_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
