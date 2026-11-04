"""Run one original TurboMPC JAX/cuDSS solve through its official Torch bridge.

This is an M0 installation check, not a numerical-gradient or performance
acceptance test. The example bridge files are imported without adding the
upstream source tree to sys.path, so the installed CUDA-enabled package is used.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path


def load_example(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Preserve the existing check result: {args.output}")
    report = {
        "scope": "M0 original JAX/Torch bridge smoke check; not M4 acceptance",
        "time_utc": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "executable": sys.executable,
        "expected_upstream_commit": "7050a4f367d08504ecb6f5820d9161e704df5816",
        "status": "failed",
    }
    try:
        import jax
        import jax.numpy as jnp
        import numpy as np
        import torch
        import turbompc
        from turbompc.dynamics.integrators import get_state_trajectory
        from turbompc.dynamics.linear_dynamics import LinearDynamics
        from turbompc.problems.optimal_control_problem import OptimalControlProblem
        from turbompc.solvers.status import SQPStatus
        from turbompc.utils.load_params import load_problem_params

        if jax.__version__ != "0.6.2" or not jax.config.jax_enable_x64:
            raise RuntimeError("The original baseline requires JAX 0.6.2 with x64 enabled")
        if not torch.cuda.is_available() or not torch.version.cuda.startswith("12."):
            raise RuntimeError("The original baseline requires the Torch CUDA 12 profile")
        devices = jax.devices()
        if not devices or any(device.platform not in {"gpu", "cuda"} for device in devices):
            raise RuntimeError(f"The baseline must use GPU JAX devices: {devices}")
        package_path = Path(turbompc.__file__).resolve()
        if not package_path.is_relative_to(Path(sys.prefix).resolve()):
            raise RuntimeError(f"TurboMPC must be loaded from the selected venv: {package_path}")
        source = Path(os.environ["TURBOMPC_BASELINE_SOURCE"]).resolve()
        bridge_path = source / "examples/rl/util/jax_torch.py"
        backend_path = source / "examples/rl/util/mpc.py"
        bridge_module = load_example("_turbompc_original_torch_bridge", bridge_path)
        backend_module = load_example("_turbompc_original_mpc_backend", backend_path)
        report.update({
            "versions": {name: importlib.metadata.version(name) for name in (
                "jax", "jaxlib", "torch", "turbompc", "nvidia-cudss-cu12",
            )},
            "gpu": torch.cuda.get_device_name(0),
            "jax_devices": [str(device) for device in devices],
            "installed_turbompc": str(package_path),
            "official_bridge_sources": {
                str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (bridge_path, backend_path)
            },
        })

        class LinearProblem:
            N, nx, nu, cost_dim = 15, 4, 2, 10
            constrain_initial_control = False
            weight_keys = (
                "weights_penalization_reference_state_trajectory",
                "weights_penalization_final_state",
                "weights_penalization_control_squared",
            )

            def __init__(self):
                self.dynamics = LinearDynamics()
                self.base = load_problem_params("linear.yaml")
                self.base.update({
                    "horizon": self.N,
                    "discretization_resolution": 0.1,
                    "discretization_scheme": 0,
                    "initial_state": jnp.zeros(self.nx),
                    "initial_guess_final_state": jnp.zeros(self.nx),
                    "reference_state_trajectory": jnp.zeros((self.N + 1, self.nx)),
                    "reference_control_trajectory": jnp.zeros((self.N + 1, self.nu)),
                    self.weight_keys[0]: jnp.array([1.0, 1.0, 0.2, 0.2]),
                    self.weight_keys[1]: jnp.array([2.0, 2.0, 0.2, 0.2]),
                    self.weight_keys[2]: jnp.array([0.2, 0.3]),
                    "state_min_bounds": jnp.full(self.nx, -5.0),
                    "state_max_bounds": jnp.full(self.nx, 5.0),
                    "control_min_bounds": jnp.full(self.nu, -2.0),
                    "control_max_bounds": jnp.full(self.nu, 2.0),
                    "use_slack_variables": False,
                })

            def make_ocp(self):
                return OptimalControlProblem(self.dynamics, self.base)

            def params_from_context(self, context):
                return {**self.base, "initial_state": context}, None

            def cost_weights(self, logits, context=None):
                factors = jnp.split(jnp.exp(logits), (self.nx, 2 * self.nx), axis=-1)
                return {key: self.base[key] * factor for key, factor in zip(self.weight_keys, factors, strict=True)}

            def clip_control(self, controls):
                return jnp.clip(controls, self.base["control_min_bounds"], self.base["control_max_bounds"])

            @staticmethod
            def mpc_warm_start_state(predicted_state, context):
                return context

            def build_mpc_inputs(self, costs, context):
                controls = jnp.zeros((context.shape[0], self.N + 1, self.nu))
                states = jax.vmap(lambda state, plan: get_state_trajectory(
                    self.dynamics, self.base["discretization_resolution"], 0,
                    self.base["dynamics_state_dot_params"], state, plan,
                ))(context, controls)
                return self.params_from_context(context)[0], self.cost_weights(costs), states, controls

        problem = LinearProblem()
        solver_config = {
            "forward_backend": "admm_fused_cudss",
            "backward_backend": "direct_cudss_ffi",
            "constraint_handling": "projection",
            "tol_convergence": 1e-7,
            "num_sqp_iteration_max": 10,
            "admm": {"max_iter": 1000, "eps_abs": 1e-9, "eps_rel": 1e-9},
        }
        report["problem"] = {"horizon": problem.N, "batch": 1, "nx": problem.nx, "nu": problem.nu, "dt": 0.1}
        report["solver_overrides"] = solver_config
        backend = backend_module.MPCBackend({"solver": solver_config}, problem)
        mpc = bridge_module.MPC(backend)
        costs = torch.zeros((1, problem.cost_dim), device="cuda", dtype=torch.float64, requires_grad=True)
        context = torch.tensor([[0.2, -0.1, 0.05, -0.02]], device="cuda", dtype=torch.float64, requires_grad=True)
        callbacks = {}
        report["gradient_callbacks"] = callbacks

        def check_callback(name, gradient):
            callbacks[name] = gradient.detach().cpu().tolist()
            if not bool(torch.isfinite(gradient).all()):
                raise RuntimeError(f"Non-finite official bridge gradient callback: {name}")

        mean = mpc.mean_only(costs, context, backend.zero_seed(1), grad_callback=check_callback)
        result = mpc.pop_result()
        report["solver_stats"] = {key: np.asarray(value).tolist() for key, value in result.stats.items()}
        report["valid"] = np.asarray(result.valid).tolist()
        report["u0"] = mean.detach().cpu().tolist()
        statuses = np.asarray(result.stats["status"])
        if not np.all(statuses == int(SQPStatus.SOLVED)) or not np.all(np.asarray(result.valid)):
            raise RuntimeError(f"The smoke check requires SQPStatus.SOLVED: {statuses.tolist()}")

        states = np.asarray(result.iterate.states)
        controls = np.asarray(result.iterate.controls)
        context_cpu = context.detach().cpu().numpy()
        dynamics = problem.base["dynamics_state_dot_params"]
        derivatives = states[:, :-1] @ np.asarray(dynamics["A"]).T + controls[:, :-1] @ np.asarray(dynamics["B"]).T
        equality = max(
            float(np.max(np.abs(states[:, 0] - context_cpu))),
            float(np.max(np.abs(states[:, 1:] - states[:, :-1] - 0.1 * derivatives))),
        )
        inequality = max(0.0, float(np.max(np.abs(states) - 5.0)), float(np.max(np.abs(controls) - 2.0)))
        report["original_unit_residuals"] = {"equality_max_abs": equality, "bound_max_violation": inequality}
        if not np.isfinite(equality + inequality) or max(equality, inequality) > 1e-5:
            raise RuntimeError("Original-unit dynamics or bound violation exceeds 1e-5")

        # MPCBackend sanitizes its VJP before the bridge callback. Check the
        # original retained pullback once so partial NaNs cannot pass as zeros.
        du = jnp.zeros_like(result.controls).at[:, 0].set(2.0 * jnp.asarray(mean.detach().cpu().numpy()))
        raw_pullback = jax.jit(jax.vmap(lambda pullback, dx, dcontrol: pullback((dx, dcontrol))))
        raw_cost_gradient, raw_context_gradient = raw_pullback(result.cache.pullbacks, jnp.zeros_like(result.states), du)
        raw_cost_gradient = np.asarray(raw_cost_gradient)
        raw_context_gradient = np.asarray(raw_context_gradient)
        report["raw_vjp_finite"] = bool(np.isfinite(raw_cost_gradient).all() and np.isfinite(raw_context_gradient).all())
        report["raw_cost_gradient"] = raw_cost_gradient.tolist()
        if not report["raw_vjp_finite"]:
            raise RuntimeError("Original retained VJP is non-finite before upstream sanitization")

        mean.square().sum().backward()
        torch.cuda.synchronize()
        report["torch_cost_gradient"] = costs.grad.detach().cpu().tolist() if costs.grad is not None else None
        report["torch_context_gradient"] = context.grad.detach().cpu().tolist() if context.grad is not None else None
        if set(callbacks) != {"mpc_cost_adjoint", "mpc_context_adjoint"}:
            raise RuntimeError("The official bridge did not report both gradient callbacks")
        if costs.grad is None or not bool(torch.isfinite(costs.grad).all()) or float(costs.grad.norm()) == 0.0:
            raise RuntimeError("Torch must receive a finite nonzero cost gradient")
        paths = sorted({
            line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
            if len(line.split()) >= 6 and ("_ffi.so" in line or any(name in line for name in (
                "libcudss", "libcudart", "libcublas", "libnvrtc", "libcusparse",
            )))
        })
        report["loaded_solver_libraries"] = paths
        required_ffi = {"libadmm_cudss_ffi.so", "libcudss_sparse_kkt_ffi.so"}
        ffi_paths = [Path(path) for path in paths if Path(path).name in required_ffi]
        if {path.name for path in ffi_paths} != required_ffi:
            raise RuntimeError("The requested forward and backward CUDA FFI libraries were not both loaded")
        if any(not path.resolve().is_relative_to(Path(sys.prefix).resolve()) for path in ffi_paths):
            raise RuntimeError("A solver FFI library was loaded outside the selected venv")
        report["status"] = "passed"
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
