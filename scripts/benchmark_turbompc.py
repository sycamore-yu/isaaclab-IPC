"""Native Torch vs pinned JAX vs unchanged official Torch/JAX bridge.

Run each backend in its isolated environment. Compilation and cold start are
reported separately. All timings include device completion, and F+B includes
the same weighted control-output loss and gradients of costs/initial state.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from turbompc_benchmark_common import UPSTREAM_COMMIT, make_case, physical_residuals


def options():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("torch", "jax", "bridge"), required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--nonlinear", action="store_true")
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=Path("build/reference"))
    parser.add_argument("--finite-difference", action="store_true")
    parser.add_argument("--compile", action="store_true", help="Compile native tensor algebra with torch.compile")
    parser.add_argument(
        "--backward-backend",
        default="direct_cudss_ffi",
        choices=("direct_cudss_ffi", "direct_jax_dense"),
        help="JAX backward implementation; dense is an accuracy diagnostic, not the primary timing baseline",
    )
    return parser.parse_args()


def gpu_snapshot():
    return subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()


def prepare_torch(case, args):
    import torch

    from isaaclab_ipc.turbompc import (
        ADMMConfig,
        AutoDiffDynamics,
        LinearDynamics,
        MPCInputs,
        SQPConfig,
        TorchMPC,
        rk4_step,
    )

    kw = dict(device="cuda:0", dtype=torch.float64)
    nx, nu, horizon = case["nx"], case["nu"], case["horizon"]
    dt = case["dt"]
    if args.nonlinear:

        def rhs(x, u):
            return torch.stack((x[1], u[0] - 0.3 * x[0].sin() - 0.1 * x[1]))

        dynamics = AutoDiffDynamics(lambda z: rk4_step(rhs, z, nx, dt), nx=nx)
    else:
        matrix = torch.zeros((nx, nx + nu), **kw)
        matrix[:, :nx] = torch.eye(nx, **kw)
        for axis in range(3):
            matrix[axis, axis + 3] = dt
            matrix[axis, axis + 6] = dt**2 / 2
            matrix[axis, axis + 9] = dt**3 / 6
            matrix[axis + 3, axis + 6] = dt
            matrix[axis + 3, axis + 9] = dt**2 / 2
            matrix[axis + 6, axis + 9] = dt
        dynamics = LinearDynamics(matrix, nx=nx)
    solver = TorchMPC(
        dynamics,
        SQPConfig(max_iter=15, tolerance=1e-6, linesearch="merit", compile_kernels=args.compile),
        ADMMConfig(max_iter=4000, eps_abs=1e-8, eps_rel=1e-8),
    )
    values = {k: torch.as_tensor(v, **kw) for k, v in case.items() if isinstance(v, np.ndarray)}
    costs = values["costs"].requires_grad_()
    x0 = values["x0"].requires_grad_()
    ref = torch.cat((values["reference"], torch.zeros(nu, **kw)))
    bounds = torch.tensor([case["state_limit"]] * nx + [case["control_limit"]] * nu, **kw)
    g = torch.eye(nx + nu, **kw).expand(args.batch, horizon + 1, nx + nu, nx + nu)
    e = torch.zeros((args.batch, horizon, nx + nu, nx + nu), **kw)

    def forward(c=costs, x=x0):
        weights = values["diagonal"][None] * torch.exp(c)
        d = torch.diag_embed(weights).unsqueeze(1).expand(-1, horizon + 1, -1, -1)
        q = (-weights * ref).unsqueeze(1).expand(-1, horizon + 1, -1)
        inputs = MPCInputs(
            x, d, e, q, g, -bounds.expand(args.batch, horizon + 1, -1), bounds.expand(args.batch, horizon + 1, -1)
        )
        return solver.solve(inputs, values["initial"])

    def full():
        result = forward()
        loss = (result.controls * values["probe"]).sum()
        gradients = torch.autograd.grad(loss, (costs, x0))
        return result, gradients

    def export(result):
        solution, gradients = result
        return {
            "trajectory": solution.trajectory.detach().cpu().numpy(),
            "cost_gradient": gradients[0].detach().cpu().numpy(),
            "state_gradient": gradients[1].detach().cpu().numpy(),
            "status": solution.status.cpu().numpy(),
            "iterations": solution.iterations.cpu().numpy(),
            "admm_iterations": solution.admm.iterations.cpu().numpy(),
            "errors": solution.errors.cpu().numpy(),
        }

    def fd(direction, step):
        return float(((forward(costs.detach() + step * direction, x0).controls * values["probe"]).sum()).detach())

    return SimpleNamespace(
        forward=forward,
        full=full,
        export=export,
        ready=lambda value: torch.cuda.synchronize(),
        close=solver.close,
        costs=costs,
        fd=fd,
        tensor=lambda x: torch.as_tensor(x, **kw),
        version=dict(torch=torch.__version__, cuda=torch.version.cuda),
        memory=lambda: dict(allocated=torch.cuda.max_memory_allocated(), reserved=torch.cuda.max_memory_reserved()),
    )


def prepare_jax(case, args):
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    ref_root = args.reference.resolve()
    actual = subprocess.check_output(["git", "-C", str(ref_root), "rev-parse", "HEAD"], text=True).strip()
    if actual != UPSTREAM_COMMIT:
        raise RuntimeError(f"Wrong upstream commit: {actual}")
    # Import the installed package, not a source tree that may shadow its .so files.
    from turbompc.dynamics.base_dynamics import Dynamics
    from turbompc.problems.optimal_control_problem import OptimalControlProblem
    from turbompc.solvers.turbompc_solver import TurboMPCSolver
    from turbompc.utils.load_params import load_problem_params, load_solver_params

    nx, nu, horizon = case["nx"], case["nu"], case["horizon"]

    class Model(Dynamics):
        def state_dot(self, x, u, params):
            if args.nonlinear:
                return jnp.stack((x[1], u[0] - 0.3 * jnp.sin(x[0]) - 0.1 * x[1]))
            return jnp.concatenate((x[3:6], x[6:9], u))

    model = Model(
        dict(
            num_states=nx,
            num_controls=nu,
            names_states=[str(i) for i in range(nx)],
            names_controls=[str(i) for i in range(nu)],
        )
    )
    params = load_problem_params("required_parameters.yaml")
    params.update(
        horizon=horizon,
        discretization_resolution=case["dt"],
        discretization_scheme=2,
        initial_state=jnp.asarray(case["x0"][0]),
        initial_guess_final_state=jnp.asarray(case["reference"]),
        initial_control=jnp.zeros(nu),
        constrain_initial_control=False,
        rescale_optimization_variables=False,
        state_rescaling_min=jnp.zeros(nx),
        state_rescaling_max=jnp.ones(nx),
        control_rescaling_min=jnp.zeros(nu),
        control_rescaling_max=jnp.ones(nu),
        reference_state_trajectory=jnp.tile(jnp.asarray(case["reference"])[None], (horizon + 1, 1)),
        reference_control_trajectory=jnp.zeros((horizon + 1, nu)),
        weights_penalization_reference_state_trajectory=jnp.asarray(case["diagonal"][:nx] / 2),
        weights_penalization_final_state=jnp.zeros(nx),
        weights_penalization_control_squared=jnp.asarray(case["diagonal"][nx:] / 2),
        weights_penalization_control_rate=jnp.zeros(nu),
        state_min_bounds=jnp.full(nx, -case["state_limit"]),
        state_max_bounds=jnp.full(nx, case["state_limit"]),
        control_min_bounds=jnp.full(nu, -case["control_limit"]),
        control_max_bounds=jnp.full(nu, case["control_limit"]),
        use_slack_variables=False,
        dynamics_state_dot_params={},
    )
    config = load_solver_params("turbompc.yaml")
    config.update(
        num_sqp_iteration_max=15,
        tol_convergence=1e-6,
        convergence_criterion="first_order",
        constraint_handling="projection",
        log_barrier_kappa=0.0,
    )
    config["linesearch"]["method"] = "merit"
    config["admm"].update(max_iter=4000, eps_abs=1e-8, eps_rel=1e-8)
    solver = TurboMPCSolver(
        OptimalControlProblem(model, params),
        config,
        forward_backend="admm_fused_cudss",
        backward_backend=args.backward_backend,
    )
    initial = solver.initial_guess(params)
    diagonal = jnp.asarray(case["diagonal"])
    costs, context = jnp.asarray(case["costs"]), jnp.asarray(case["x0"])
    probe = jnp.asarray(case["probe"])

    def one(c, x):
        weight = diagonal * jnp.exp(c) / 2
        seed = initial._replace(states=jnp.tile(x[None], (horizon + 1, 1)), controls=jnp.zeros((horizon + 1, nu)))
        solution = solver.solve(
            seed,
            params,
            {
                "initial_state": x,
                "weights_penalization_reference_state_trajectory": weight[:nx],
                "weights_penalization_control_squared": weight[nx:],
            },
        )
        return (solution.states, solution.controls), (
            solution.status,
            solution.num_iter,
            solution.convergence_error,
            solution.admm_iters,
        )

    many = jax.vmap(one)
    forward = jax.jit(many)

    def retain(c, x):
        output, pullback, aux = jax.vjp(many, c, x, has_aux=True)
        return output, pullback, aux

    retained = jax.jit(retain)
    apply_adjoint = jax.jit(lambda pullback, dx, du: pullback((dx, du)))

    def full(c=costs, x=context):
        output, pullback, aux = retained(c, x)
        gradients = apply_adjoint(pullback, jnp.zeros_like(output[0]), probe)
        return output, gradients, aux

    def ready(tree):
        return jax.tree.map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, tree)

    def export(result):
        output, gradients, aux = result
        return dict(
            trajectory=np.asarray(jnp.concatenate(output, -1)),
            cost_gradient=np.asarray(gradients[0]),
            state_gradient=np.asarray(gradients[1]),
            status=np.asarray(aux[0]),
            iterations=np.asarray(aux[1]),
            errors=np.asarray(aux[2]),
            admm_iterations=np.asarray(aux[3]),
        )

    def fd(direction, step):
        return float(jnp.sum(forward(costs + step * direction, context)[0][1] * probe))

    def close():
        # Release FFI-owned CUDA resources while the XLA client and loaded CUDA
        # libraries are still alive. The pinned solver has owner-level close()
        # methods but no top-level close; do not defer them to interpreter exit.
        jax.effects_barrier()
        for admm in (solver._admm_solver_fwd, solver._admm_solver_bwd):
            if admm is not None and admm._cudss_workspace is not None:
                admm._cudss_workspace.close()
        if solver._cudss_backward_solver is not None:
            solver._cudss_backward_solver.close()
        for schur in (solver._schur_solver_fwd, solver._schur_solver_bwd):
            if schur is not None:
                schur.close()
        jax.effects_barrier()

    if args.backend == "jax":
        return SimpleNamespace(
            forward=lambda: forward(costs, context),
            full=full,
            ready=ready,
            export=export,
            close=close,
            costs=costs,
            fd=fd,
            tensor=jnp.asarray,
            version=dict(jax=jax.__version__, upstream=actual),
            memory=lambda: dict(jax.devices()[0].memory_stats() or {}),
        )

    # Load the actual pinned official bridge file without copying or editing it.
    import torch

    bridge_file = ref_root / "examples/rl/util/jax_torch.py"
    spec = importlib.util.spec_from_file_location("official_turbompc_bridge", bridge_file)
    official = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = official
    spec.loader.exec_module(official)

    class Backend:
        supports_plan_adjoint = True

        def solve_device(self, c, x, seed, *, retain, **kwargs):
            output, pullback, aux = retained(c, x)
            return SimpleNamespace(
                mean=output[1][:, 0], states=output[0], controls=output[1], cache=pullback, valid=aux[0] == 0, stats=aux
            )

        def vjp_plan_device(self, cache, dx, du):
            dc, dx0 = apply_adjoint(cache, dx, du)
            return dict(costs=dc, context=dx0)

    bridge = official.MPC(Backend())
    c_t = torch.tensor(case["costs"], device="cuda", dtype=torch.float64, requires_grad=True)
    x_t = torch.tensor(case["x0"], device="cuda", dtype=torch.float64, requires_grad=True)
    p_t = torch.tensor(case["probe"], device="cuda", dtype=torch.float64)

    def bridge_forward():
        mean, controls = bridge(c_t, x_t, None)
        return controls, bridge.pop_result()

    def bridge_full():
        controls, result = bridge_forward()
        gradient = torch.autograd.grad((controls * p_t).sum(), (c_t, x_t))
        return result, gradient

    def bridge_export(result):
        solution, gradients = result
        return dict(
            trajectory=np.asarray(jnp.concatenate((solution.states, solution.controls), -1)),
            cost_gradient=gradients[0].detach().cpu().numpy(),
            state_gradient=gradients[1].detach().cpu().numpy(),
            status=np.asarray(solution.stats[0]),
            iterations=np.asarray(solution.stats[1]),
            errors=np.asarray(solution.stats[2]),
            admm_iterations=np.asarray(solution.stats[3]),
        )

    return SimpleNamespace(
        forward=bridge_forward,
        full=bridge_full,
        ready=lambda x: torch.cuda.synchronize(),
        export=bridge_export,
        close=close,
        costs=c_t,
        fd=fd,
        tensor=jnp.asarray,
        version=dict(
            jax=jax.__version__,
            torch=torch.__version__,
            upstream=actual,
            bridge_sha256=hashlib.sha256(bridge_file.read_bytes()).hexdigest(),
        ),
        memory=lambda: dict(torch_allocated=torch.cuda.max_memory_allocated(), jax=jax.devices()[0].memory_stats()),
    )


def main():
    args = options()
    if min(args.batch, args.horizon, args.repeats) < 1 or args.warmup < 1:
        raise ValueError("Positive batch, horizon, repeats and warmup are required")
    npz_path = args.output.with_suffix(".npz")
    if args.output == npz_path:
        raise ValueError("The JSON output must differ from its .npz sidecar")
    for path in (args.output, npz_path):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Preserve existing result: {path}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    case = make_case(args.batch, args.horizon, args.nonlinear)
    start = time.perf_counter()
    runtime = prepare_torch(case, args) if args.backend == "torch" else prepare_jax(case, args)
    cold_result = runtime.full()
    runtime.ready(cold_result)
    cold_seconds = time.perf_counter() - start
    exported = runtime.export(cold_result)
    report = dict(
        backend=args.backend,
        batch=args.batch,
        horizon=args.horizon,
        nonlinear=args.nonlinear,
        compiled=args.compile,
        version=runtime.version,
        backward_backend=args.backward_backend,
        cold_seconds=cold_seconds,
        gpu_before=gpu_snapshot(),
        physical_residuals=physical_residuals(exported["trajectory"], case, args.nonlinear),
        status=exported["status"].tolist(),
        iterations=exported["iterations"].tolist(),
        repeats=args.repeats,
        warmup=args.warmup,
    )
    for name, operation in (("forward", runtime.forward), ("forward_backward", runtime.full)):
        for _ in range(args.warmup):
            runtime.ready(operation())
        samples = []
        for _ in range(args.repeats):
            runtime.ready(None)
            t0 = time.perf_counter()
            runtime.ready(operation())
            samples.append(1000 * (time.perf_counter() - t0))
        report[name] = dict(
            median_ms=float(np.median(samples)), p95_ms=float(np.percentile(samples, 95)), samples_ms=samples
        )
    if args.finite_difference:
        rng = np.random.default_rng(45)
        direction = rng.normal(size=case["costs"].shape)
        direction /= np.linalg.norm(direction)
        analytic = float(np.sum(exported["cost_gradient"] * direction))
        tensor_direction = runtime.tensor(direction)
        h = 1e-4
        numerical = (runtime.fd(tensor_direction, h) - runtime.fd(tensor_direction, -h)) / (2 * h)
        report["finite_difference"] = dict(
            analytic=analytic,
            numerical=numerical,
            absolute_error=abs(analytic - numerical),
            relative_error=abs(analytic - numerical) / max(abs(numerical), 1e-12),
        )
    report["gpu_after"] = gpu_snapshot()
    report["memory"] = runtime.memory()
    with npz_path.open("xb") as handle:
        np.savez(handle, **exported, **{f"input_{k}": v for k, v in case.items()})
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, default=lambda value: int(value))
        handle.write("\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("memory", "status", "iterations")}, indent=2))
    runtime.close()


if __name__ == "__main__":
    main()
