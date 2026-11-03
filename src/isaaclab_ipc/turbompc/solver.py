"""Projection SQP with quadratic costs, nonlinear dynamics and stage constraints.

The forward uses TurboMPC's CUDA ADMM; backward solves the full Lagrangian KKT
system rather than differentiating through SQP/ADMM iterations. Only regular
local solutions have a unique classical derivative. Failed samples return zero
parameter gradients and retain their explicit nonzero status.
"""

from collections import OrderedDict
from dataclasses import dataclass, fields, replace

import torch
from torch.autograd.function import once_differentiable

from .adjoint import SparseAdjoint
from .constraints import PathConstraints
from .dynamics import LinearDynamics
from .native import ADMMConfig, ADMMResult, NativeADMM, QPData


@dataclass(frozen=True)
class MPCInputs:
    initial_state: torch.Tensor
    D: torch.Tensor
    E: torch.Tensor
    q: torch.Tensor
    G: torch.Tensor
    lower: torch.Tensor
    upper: torch.Tensor


@dataclass(frozen=True)
class SQPConfig:
    max_iter: int = 15
    tolerance: float = 1e-6
    linesearch: str = "none"
    alphas: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
    armijo: float = 0.4
    max_cached_shapes: int = 4
    compile_kernels: bool = False
    # Upstream permits an outer solution even when the inner QP exhausts its budget.
    require_qp_convergence: bool = False


@dataclass(frozen=True)
class MPCResult:
    trajectory: torch.Tensor
    status: torch.Tensor
    iterations: torch.Tensor
    errors: torch.Tensor
    admm: ADMMResult
    nx: int
    # errors[:, 2] remains the original physical inequality violation.
    normal_cone_error: torch.Tensor | None = None

    @property
    def states(self):
        return self.trajectory[..., : self.nx]

    @property
    def controls(self):
        return self.trajectory[..., self.nx :]


def _matvec(a, x):
    return (a @ x.unsqueeze(-1)).squeeze(-1)


def _cost_gradient(inputs, z):
    gradient = _matvec(inputs.D, z) + inputs.q
    gradient[:, 1:] += _matvec(inputs.E, z[:, :-1])
    gradient[:, :-1] += _matvec(inputs.E.transpose(-1, -2), z[:, 1:])
    return gradient


def _cost(inputs, z):
    return (0.5 * z * _matvec(inputs.D, z) + inputs.q * z).sum((-1, -2)) + (
        z[:, 1:] * _matvec(inputs.E, z[:, :-1])
    ).sum((-1, -2))


def _choose(mask, selected, previous):
    return torch.where(mask.reshape(-1, *([1] * (selected.ndim - 1))), selected, previous)


def _choose_admm(mask, selected, previous):
    if previous is None:
        return selected
    return ADMMResult(*[_choose(mask, getattr(selected, f.name), getattr(previous, f.name)) for f in fields(selected)])


def _active_sets(g, lower, upper, z, y, eps_abs, eps_rel):
    value = _matvec(g, z)
    dl, du = value - lower, upper - value
    near_upper = du < dl
    bound = torch.where(near_upper, upper, lower)
    touching = torch.minimum(dl, du) <= eps_abs + eps_rel * bound.abs()
    dual_lower, dual_upper = y < -eps_abs, y > eps_abs
    undecided = ~(dual_lower | dual_upper)
    active_lower = dual_lower | (undecided & touching & ~near_upper)
    active_upper = dual_upper | (undecided & touching & near_upper)
    active = (active_lower | active_upper) & torch.isfinite(bound) & (g.abs().amax(-1) > 0)
    return active, active_lower, active_upper


def _parameter_gradients(z, y, v, w, wg, good, active, active_lower, active_upper, nx, needs):
    v = _choose(good, v, torch.zeros_like(v))
    w = _choose(good, w, torch.zeros_like(w))
    wg = _choose(good, wg, torch.zeros_like(wg))
    grad_d = grad_e = grad_g = None
    if needs[1]:
        dz = v.unsqueeze(-1) * z.unsqueeze(-2)
        grad_d = -0.5 * (dz + dz.transpose(-1, -2))
    if needs[2]:
        grad_e = -(v[:, 1:, :, None] * z[:, :-1, None, :] + z[:, 1:, :, None] * v[:, :-1, None, :])
    if needs[4]:
        grad_g = -(y[..., None] * v[..., None, :] + wg[..., None] * z[..., None, :]) * active[..., None]
        grad_g = _choose(good, grad_g, torch.zeros_like(grad_g))
    return (
        w[:, :nx] if needs[0] else None,
        grad_d,
        grad_e,
        -v if needs[3] else None,
        grad_g,
        wg * active_lower if needs[5] else None,
        wg * active_upper if needs[6] else None,
    )


class _ImplicitTrajectory(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x0, d, e, q, g, lower, upper, trajectory, lambda0, lambda_dyn, y, status, solver):
        ctx.solver = solver
        ctx.save_for_backward(x0, d, e, q, g, lower, upper, trajectory, lambda0, lambda_dyn, y, status)
        return trajectory.clone()

    @staticmethod
    @once_differentiable
    def backward(ctx, upstream):
        x0, d, e, q, g, lower, upper, z, lambda0, lambda_dyn, y, status = ctx.saved_tensors
        inputs = MPCInputs(x0, d, e, q, g, lower, upper)
        good = status == 0
        if good.all():
            safe_inputs, z_safe, ld_safe, y_safe = inputs, z, lambda_dyn, y
        else:
            if not good.any():
                return *[torch.zeros_like(getattr(inputs, f.name)) for f in fields(inputs)], *([None] * 6)
            # Invalid samples must not create singular/NaN systems even at zero RHS.
            template = good.to(torch.int32).argmax().reshape(1)

            def safe(x):
                return _choose(good, x, x.index_select(0, template).expand_as(x))

            safe_inputs = MPCInputs(*[safe(getattr(inputs, f.name)) for f in fields(inputs)])
            z_safe, ld_safe, y_safe = safe(z), safe(lambda_dyn), safe(y)
        solver = ctx.solver
        qp, hessian, active, parameter_active, active_lower, active_upper, parameter_y, masked_upstream = (
            solver._adjoint_data(safe_inputs, z_safe, ld_safe, y_safe, good, upstream)
        )
        adjoint = solver._adjoint(qp)
        v, w, wg = adjoint.solve(qp, hessian, active, masked_upstream)
        if not (torch.isfinite(v).all() & torch.isfinite(w).all() & torch.isfinite(wg).all()):
            raise RuntimeError("Nonfinite MPC adjoint: the active KKT system may be singular")
        gradients = solver._parameter_gradients(
            z_safe,
            parameter_y,
            v,
            w,
            wg if solver.constraints is None else wg[..., : safe_inputs.G.shape[-2]],
            good,
            parameter_active,
            active_lower,
            active_upper,
            solver.dynamics.nx,
            ctx.needs_input_grad[:7],
        )
        return *gradients, *([None] * 6)


class TorchMPC:
    """Batched multiple-shooting MPC with a fixed dynamics object.

    Stage cost is 0.5*z[k].T*D[k]*z[k] + q[k].T*z[k], plus
    z[k+1].T*E[k]*z[k]. Bounds are lower <= G[k]*z[k] <= upper;
    optional fixed PathConstraints add smooth nonlinear stage bounds.
    All tensors include a batch axis and terminal control, as in TurboMPC.
    Costs must be convex in the forward QP. Only projection without relaxation
    is supported here; unsupported modes fail rather than silently softening constraints.
    """

    def __init__(
        self,
        dynamics,
        config: SQPConfig | None = None,
        admm_config: ADMMConfig | None = None,
        *,
        constraints: PathConstraints | None = None,
    ):
        self.dynamics = dynamics
        self.constraints = constraints
        self.config = config or SQPConfig()
        self.admm = NativeADMM(admm_config)
        if self.config.max_iter < 1 or self.config.tolerance <= 0:
            raise ValueError("Invalid SQP iteration budget or tolerance")
        if self.config.linesearch not in ("none", "merit"):
            raise ValueError("This port supports upstream none/merit line search")
        if self.admm.config.constraint_handling != 0 or self.admm.config.use_slack:
            raise ValueError("TorchMPC currently requires projection constraints without slack")
        self._adjoints = OrderedDict()
        self._dynamics_hessian = self.dynamics.lagrangian_hessian
        self._active_sets = _active_sets
        self._parameter_gradients = _parameter_gradients
        if constraints is not None:
            self._constraint_hessian = constraints.lagrangian_hessian
            self._path_active_sets = constraints.active_sets
            self._path_normal_cone = constraints.normal_cone_residual
        if self.config.compile_kernels:
            # Compile only tensor algebra. cuDSS workspaces and convergence
            # decisions remain outside graph capture; saved outputs stay owned.
            self._qp = torch.compile(self._qp, fullgraph=True, dynamic=False)
            self._errors = torch.compile(self._errors, fullgraph=True, dynamic=False)
            self._line_search = torch.compile(self._line_search, fullgraph=True, dynamic=False)
            self._advance = torch.compile(self._advance, fullgraph=True, dynamic=False)
            self._adjoint_data = torch.compile(self._adjoint_data, fullgraph=True, dynamic=False)
            self.admm._schur = torch.compile(self.admm._schur, fullgraph=True, dynamic=False)
            self._dynamics_hessian = torch.compile(self._dynamics_hessian, fullgraph=True, dynamic=False)
            self._active_sets = torch.compile(self._active_sets, fullgraph=True, dynamic=False)
            self._parameter_gradients = torch.compile(self._parameter_gradients, fullgraph=True, dynamic=False)
            if constraints is not None:
                self._constraint_hessian = torch.compile(self._constraint_hessian, fullgraph=True, dynamic=False)
                self._path_active_sets = torch.compile(self._path_active_sets, fullgraph=True, dynamic=False)
                self._path_normal_cone = torch.compile(self._path_normal_cone, fullgraph=True, dynamic=False)

    def _qp(self, inputs, z):
        b, t, n = z.shape
        nx = self.dynamics.nx
        if type(self.dynamics) is LinearDynamics:
            # For f(z)=M z+b, the equality is M z[k]-x[k+1]=-b.
            # Reuse the exact Jacobian and offset instead of two cancelling
            # matrix products at every SQP expansion and backward preparation.
            jacobian = self.dynamics.matrix.expand(b, t - 1, nx, n)
            c_dyn = z.new_zeros((b, t - 1, nx))
            if self.dynamics.offset is not None:
                # Preserve Torch's original dtype promotion and broadcasting,
                # including a CPU scalar offset for CUDA trajectories.
                c_dyn = c_dyn - self.dynamics.offset
        else:
            value, jacobian = self.dynamics.linearize(z[:, :-1])
            c_dyn = _matvec(jacobian, z[:, :-1]) - value
        eye = torch.eye(nx, n, device=z.device, dtype=z.dtype)
        g, lower, upper = inputs.G, inputs.lower, inputs.upper
        if self.constraints is not None:
            path_value, path_jacobian = self.constraints.linearize(z)
            enabled = self.constraints.enabled(z)
            path_jacobian = torch.where(enabled[..., None], path_jacobian, 0.0)
            offset = torch.where(enabled, path_value - _matvec(path_jacobian, z), 0.0)
            path_lower, path_upper = self.constraints.bounds(z)
            g = torch.cat((g, path_jacobian), -2)
            lower = torch.cat((lower, path_lower - offset), -1)
            upper = torch.cat((upper, path_upper - offset), -1)
        return QPData(
            inputs.D,
            inputs.E,
            inputs.q,
            eye.expand(b, nx, n),
            jacobian,
            -eye.expand(b, t - 1, nx, n),
            inputs.initial_state,
            c_dyn,
            g,
            lower,
            upper,
        )

    def _violations(self, inputs, z):
        eq0 = z[:, 0, : self.dynamics.nx] - inputs.initial_state
        eq = self.dynamics(z[:, :-1]) - z[:, 1:, : self.dynamics.nx]
        gx = _matvec(inputs.G, z)
        violation = torch.maximum(inputs.lower - gx, gx - inputs.upper).clamp_min(0)
        if self.constraints is not None:
            violation = torch.cat((violation, self.constraints.violations(z)), -1)
        return torch.cat((eq0, eq.flatten(1)), -1), violation

    def _errors(self, inputs, qp, z, admm):
        stationarity = _cost_gradient(inputs, z) + _matvec(qp.G.transpose(-1, -2), admm.y)
        stationarity[:, 0] += _matvec(qp.A0.transpose(-1, -2), admm.lambda0)
        stationarity[:, :-1] += _matvec(qp.A_minus.transpose(-1, -2), admm.lambda_dyn)
        stationarity[:, 1:] += _matvec(qp.A_plus.transpose(-1, -2), admm.lambda_dyn)
        eq, violation = self._violations(inputs, z)
        inequality = violation.flatten(1).amax(-1) if violation.numel() else z.new_zeros(z.shape[0])
        return torch.stack((stationarity.flatten(1).abs().amax(-1), eq.abs().amax(-1), inequality), -1)

    def _line_search(self, inputs, current, candidate):
        if self.config.linesearch == "none":
            return candidate
        # Port of upstream backtracking_merit_step: largest accepted alpha,
        # otherwise the candidate with lowest finite merit, without extra retries.
        delta = candidate - current
        objective = _cost(inputs, current)
        derivative = (_cost_gradient(inputs, current) * delta).sum((-1, -2))
        eq, violation = self._violations(inputs, current)
        norm = eq.abs().sum(-1) + violation.sum((-1, -2))
        mu = torch.nan_to_num(derivative / (0.5 * norm.clamp_min(1e-8)), nan=0.1, posinf=1e6, neginf=0.1).clamp(
            0.1, 1e6
        )
        alphas = current.new_tensor(self.config.alphas)

        def evaluate(alpha):
            trial = current + alpha * delta
            trial_eq, trial_ineq = self._violations(inputs, trial)
            return _cost(inputs, trial) + mu * (trial_eq.abs().sum(-1) + trial_ineq.sum((-1, -2)))

        merits = torch.vmap(evaluate)(alphas).transpose(0, 1)
        decrease = merits - (
            objective[:, None]
            + mu[:, None] * norm[:, None]
            + self.config.armijo * alphas[None] * (derivative - mu * norm)[:, None]
        )
        finite = torch.isfinite(merits) & torch.isfinite(decrease)
        accepted = finite & (decrease <= 0)
        indexes = torch.arange(len(self.config.alphas), device=current.device)
        largest = torch.where(accepted, indexes[None], -1).amax(-1)
        fallback = torch.where(finite, merits, torch.inf).argmin(-1)
        selected = torch.where(largest >= 0, largest, fallback)
        return current + alphas[selected, None, None] * delta

    def _advance(self, inputs, z, admm, candidate, status, iterations, errors, normal_cone):
        """Accept one QP and check the real problem in one tensor-only region.

        The CUDA solve and host termination decision stay outside compilation.
        Keeping selection, residuals and status together avoids launching each
        batch mask separately while retaining the upstream warm-start buffers.
        """
        running = status == -1
        fatal = candidate.status >= 2
        next_z = self._line_search(inputs, z, candidate.x)
        accept = running & ~fatal
        z = _choose(accept, next_z, z)
        admm = _choose_admm(accept, candidate, admm)
        # Re-evaluate real nonlinear dynamics and stationarity at acceptance,
        # including the budget limit: a stale residual must not certify a plan.
        qp = self._qp(inputs, z)
        errors = _choose(running, self._errors(inputs, qp, z, admm), errors)
        iterations = iterations + running.to(iterations.dtype)
        finite = torch.isfinite(z).flatten(1).all(-1) & torch.isfinite(errors).all(-1)
        if self.constraints is not None:
            path_y = admm.y[..., inputs.G.shape[-2] :]
            next_normal_cone = self._path_normal_cone(z, path_y).flatten(1).amax(-1)
            normal_cone = _choose(running, next_normal_cone, normal_cone)
            finite = finite & torch.isfinite(normal_cone)
        solved = running & ~fatal & finite & (errors.amax(-1) <= self.config.tolerance)
        if self.config.require_qp_convergence:
            solved = solved & (candidate.status == 0)
        if self.constraints is not None:
            solved = solved & (normal_cone <= self.config.tolerance)
        status = torch.where(solved, 0, status)
        status = torch.where(running & fatal, 2, status)
        status = torch.where(running & (~finite | (candidate.status == 4)), 3, status)
        return z, admm, qp, status, iterations, errors, normal_cone, (status != -1).all()

    def _adjoint_data(self, inputs, z, lambda_dyn, y, good, upstream):
        """Assemble exact curvature and active rows before the native KKT solve."""
        qp = self._qp(inputs, z)
        # This exact class has zero curvature. Keep its cost storage read-only;
        # nonlinear models and subclasses retain their full dynamics Hessian.
        if type(self.dynamics) is LinearDynamics:
            hessian = qp.D
        else:
            hessian = qp.D.clone()
            hessian[:, :-1] += self._dynamics_hessian(z[:, :-1], lambda_dyn)
        parameter_y = y
        if self.constraints is None:
            active, active_lower, active_upper = self._active_sets(
                qp.G, qp.lower, qp.upper, z, y, self.admm.config.eps_abs, self.admm.config.eps_rel
            )
            parameter_active = active
        else:
            linear_rows = inputs.G.shape[-2]
            parameter_y, path_y = y[..., :linear_rows], y[..., linear_rows:]
            parameter_active, active_lower, active_upper = self._active_sets(
                inputs.G,
                inputs.lower,
                inputs.upper,
                z,
                parameter_y,
                self.admm.config.eps_abs,
                self.admm.config.eps_rel,
            )
            path_active, _, _ = self._path_active_sets(
                z, path_y, max(self.admm.config.eps_abs, self.config.tolerance), self.admm.config.eps_rel
            )
            # Do not mutate qp.D when affine dynamics share its storage. An
            # inactive multiplier cannot contribute constraint curvature.
            hessian = hessian + self._constraint_hessian(z, torch.where(path_active, path_y, 0.0))
            active = torch.cat((parameter_active, path_active), -1)
        masked_upstream = _choose(good, upstream, torch.zeros_like(upstream))
        return qp, hessian, active, parameter_active, active_lower, active_upper, parameter_y, masked_upstream

    def _adjoint(self, qp):
        b, t, n = qp.q.shape
        key = (b, t, n, self.dynamics.nx, qp.G.shape[-2], qp.q.device)
        if key not in self._adjoints:
            self._adjoints[key] = SparseAdjoint(*key[:-1], key[-1], compile_kernels=self.config.compile_kernels)
            if len(self._adjoints) > self.config.max_cached_shapes:
                _, old = self._adjoints.popitem(last=False)
                old.close()
        self._adjoints.move_to_end(key)
        return self._adjoints[key]

    def solve(self, inputs: MPCInputs, initial_guess: MPCResult | torch.Tensor | None = None):
        if inputs.q.ndim != 3 or inputs.q.shape[-1] <= self.dynamics.nx:
            raise ValueError("Expected q[B,N+1,nx+nu]")
        inputs = replace(inputs, D=0.5 * (inputs.D + inputs.D.transpose(-1, -2)))
        with torch.inference_mode(False), torch.no_grad():
            # SQP is a numerical forward; derivatives enter only through the
            # custom implicit function below, which keeps the original inputs.
            # Detaching here also gives training and finite-difference calls
            # the same compilation metadata instead of duplicate graph variants.
            forward_inputs = MPCInputs(*[getattr(inputs, f.name).detach() for f in fields(inputs)])
            b, t, n = inputs.q.shape
            if initial_guess is None:
                z = inputs.q.new_zeros((b, t, n))
                z[..., : self.dynamics.nx] = inputs.initial_state[:, None]
                admm = None
            elif isinstance(initial_guess, MPCResult):
                z, admm = initial_guess.trajectory.detach().clone(), initial_guess.admm
            else:
                z, admm = initial_guess.detach().clone(), None
            if z.shape != inputs.q.shape:
                raise ValueError("Initial trajectory shape does not match q")
            status = torch.full((b,), -1, dtype=torch.int32, device=z.device)
            iterations = torch.zeros_like(status)
            errors = z.new_full((b, 3), torch.inf)
            normal_cone = z.new_full((b,), torch.inf) if self.constraints is not None else None
            qp = self._qp(forward_inputs, z)
            for _ in range(self.config.max_iter):
                candidate = self.admm.solve(qp, admm)
                z, admm, qp, status, iterations, errors, normal_cone, done = self._advance(
                    forward_inputs, z, admm, candidate, status, iterations, errors, normal_cone
                )
                if done:
                    break
            status = torch.where(status == -1, 1, status)
        trajectory = _ImplicitTrajectory.apply(
            inputs.initial_state,
            inputs.D,
            inputs.E,
            inputs.q,
            inputs.G,
            inputs.lower,
            inputs.upper,
            z,
            admm.lambda0,
            admm.lambda_dyn,
            admm.y,
            status,
            self,
        )
        return MPCResult(trajectory, status, iterations, errors, admm, self.dynamics.nx, normal_cone)

    def close(self):
        self.admm.close()
        for value in self._adjoints.values():
            value.close()
        self._adjoints.clear()
