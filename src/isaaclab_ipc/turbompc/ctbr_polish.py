"""Bounded CTBR KKT refinement on the unchanged physical problem.

Only cost parameters D/E/q are differentiable on this path. The initial state
and physical constraints must be fixed: a first-position face can be redundant
with the initial-state/dynamics equalities, so its individual physical-parameter
multiplier is not a unique derivative. All solves use the owner's native
SparseAdjoint/cuDSS workspace. No CPU numerical solver or primal regularizer is
used here.
"""

import math
from dataclasses import dataclass, fields, replace

import torch

from .quadrotor import CTBRDynamics

POLISH_ERROR_NAMES = (
    "stationarity",
    "equality",
    "inequality",
    "normal_cone",
    "exact_rollout_violation",
    "shooting_gap",
)

POLISH_FAILURE_REASONS = {
    0: "accepted",
    1: "not_attempted",
    2: "nonfinite_or_invalid_problem_data",
    3: "control_independent_first_position_infeasible",
    4: "native_kkt_or_tensor_operation_error",
    5: "original_kkt_residual_failed",
    6: "bounded_newton_original_problem_certificate_failed",
}


@dataclass(frozen=True)
class PolishedSolution:
    """Refined primal/dual data, separate from the untouched ADMM result.

    status: 0 accepted, 1 not attempted, 2 failed. Failed/not-attempted rows
    retain the original trajectory and multipliers and have a false active
    mask. iterations counts attempted Newton solves. errors has the six
    infinity norms named by POLISH_ERROR_NAMES; for failures it describes the
    last evaluated candidate, and for unattempted rows it is infinity.
    The rollout uses the real initial state and the returned control sequence;
    shooting_gap compares its states to the multiple-shooting trajectory.
    linear_residuals[B,max_iter,3] retains the final original KKT primal/
    equality/inequality residual at every attempted Newton step. The history
    [B,max_iter,max_linear_refinements+1,3] includes the initial solve and each
    correction, with infinity for unavailable entries. linear_refinement_steps
    [B,max_iter] counts additional attempted native solves, including errors.
    failure_reason uses POLISH_FAILURE_REASONS; prefix_rows_removed counts
    active first-position rows explicitly excluded from the KKT basis.
    """

    trajectory: torch.Tensor
    lambda0: torch.Tensor
    lambda_dyn: torch.Tensor
    y: torch.Tensor
    active_mask: torch.Tensor
    status: torch.Tensor
    iterations: torch.Tensor
    errors: torch.Tensor
    linear_residuals: torch.Tensor | None = None
    failure_reason: torch.Tensor | None = None
    prefix_rows_removed: torch.Tensor | None = None
    linear_residual_history: torch.Tensor | None = None
    linear_refinement_steps: torch.Tensor | None = None


def _mv(matrix, vector):
    return (matrix @ vector.unsqueeze(-1)).squeeze(-1)


def _maximum(value):
    return value.reshape(value.shape[0], -1).abs().amax(-1)


def _cost_gradient(inputs, z):
    gradient = _mv(inputs.D, z) + inputs.q
    gradient[:, 1:] += _mv(inputs.E, z[:, :-1])
    gradient[:, :-1] += _mv(inputs.E.transpose(-1, -2), z[:, 1:])
    return gradient


def _stationarity(inputs, qp, z, lambda0, lambda_dyn, y):
    residual = _cost_gradient(inputs, z) + _mv(qp.G.transpose(-1, -2), y)
    residual[:, 0] += _mv(qp.A0.transpose(-1, -2), lambda0)
    residual[:, :-1] += _mv(qp.A_minus.transpose(-1, -2), lambda_dyn)
    residual[:, 1:] += _mv(qp.A_plus.transpose(-1, -2), lambda_dyn)
    return residual


def _equalities(solver, inputs, z):
    initial = z[:, 0, :10] - inputs.initial_state
    dynamics = solver.dynamics(z[:, :-1]) - z[:, 1:, :10]
    return torch.cat((initial, dynamics.flatten(1)), -1)


def _physical_constraints(solver, inputs, z):
    value, lower, upper = _mv(inputs.G, z), inputs.lower, inputs.upper
    if solver.constraints is not None:
        path_lower, path_upper = solver.constraints.bounds(z)
        value = torch.cat((value, solver.constraints(z)), -1)
        lower = torch.cat((lower, path_lower), -1)
        upper = torch.cat((upper, path_upper), -1)
    return value, lower, upper


def _violation(value, lower, upper):
    return torch.maximum(lower - value, value - upper).clamp_min(0)


def _errors(solver, inputs, qp, z, lambda0, lambda_dyn, y):
    value, lower, upper = _physical_constraints(solver, inputs, z)
    normal = value - torch.clamp(value + y, min=lower, max=upper)
    state = inputs.initial_state
    rollout = [state]
    for k in range(z.shape[1] - 1):
        state = solver.dynamics(torch.cat((state, z[:, k, 10:]), -1))
        rollout.append(state)
    rollout_states = torch.stack(rollout, 1)
    exact_z = torch.cat((rollout_states, z[..., 10:]), -1)
    exact_value, exact_lower, exact_upper = _physical_constraints(solver, inputs, exact_z)
    return torch.stack(
        (
            _maximum(_stationarity(inputs, qp, z, lambda0, lambda_dyn, y)),
            _maximum(_equalities(solver, inputs, z)),
            _maximum(_violation(value, lower, upper)),
            _maximum(normal),
            _maximum(_violation(exact_value, exact_lower, exact_upper)),
            _maximum(rollout_states - z[..., :10]),
        ),
        -1,
    )


def _linear_defect(qp, hessian, active, dz, dlambda, dy, primal_rhs, equality_rhs, inequality_rhs):
    """Return b-K*delta for the unchanged original KKT matrix and RHS."""
    d0, ddyn = dlambda[:, :10], dlambda[:, 10:].reshape(dz.shape[0], dz.shape[1] - 1, 10)
    active_g = qp.G * active[..., None]
    primal = _mv(hessian, dz)
    primal[:, 1:] += _mv(qp.E, dz[:, :-1])
    primal[:, :-1] += _mv(qp.E.transpose(-1, -2), dz[:, 1:])
    primal[:, 0] += _mv(qp.A0.transpose(-1, -2), d0)
    primal[:, :-1] += _mv(qp.A_minus.transpose(-1, -2), ddyn)
    primal[:, 1:] += _mv(qp.A_plus.transpose(-1, -2), ddyn)
    primal += _mv(active_g.transpose(-1, -2), dy)
    equality = torch.cat(
        (_mv(qp.A0, dz[:, 0]), (_mv(qp.A_minus, dz[:, :-1]) + _mv(qp.A_plus, dz[:, 1:])).flatten(1)),
        -1,
    )
    # Inactive multiplier equations have an identity diagonal in SparseAdjoint.
    inequality = _mv(active_g, dz) + torch.where(active, 0.0, dy)
    return primal_rhs - primal, equality_rhs - equality, inequality_rhs - inequality


def _linear_residual(qp, hessian, active, dz, dlambda, dy, primal_rhs, equality_rhs, inequality_rhs):
    """Check the ORIGINAL KKT equations, not a factorizer's internal residual."""
    defect = _linear_defect(qp, hessian, active, dz, dlambda, dy, primal_rhs, equality_rhs, inequality_rhs)
    return torch.stack(tuple(_maximum(part) for part in defect), -1)


class CTBRPolisher:
    """At most max_iter Newton steps per already accepted CTBR sample.

    Each eligible sample uses a separate B=1 native solve so a singular or
    inaccurate KKT system cannot poison another batch row. The local active
    set is fixed during refinement. Generic dependent rows are not removed or
    regularized: a cuDSS error or a failed original-system residual certificate
    rejects that row. Only the analytically certified first-position rows are
    excluded, and every original inequality is still checked after refinement.
    Each Newton step may use up to max_linear_refinements extra solves of the
    same original matrix to correct b-K*delta. These are additional native
    numerical factorizations, not an increase to the Newton or ADMM budget.
    Neither the matrix nor its original RHS changes during these corrections.

    Adjoint solves use the fixed upstream infinity norm for RHS homogeneity,
    then certify the original KKT residual after at most four corrections.
    Normalization alone is not a precision guarantee. Detached diagnostics
    describe only the last adjoint call, not historical MPCResult objects.
    """

    adjoint_tolerance = 1e-11

    def __init__(self, max_iter=4, tolerance=1e-11, max_linear_refinements=2, max_adjoint_refinements=4):
        if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 1:
            raise ValueError("CTBR polishing needs a positive integer iteration budget")
        if not math.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("CTBR polishing needs a finite positive tolerance")
        if (
            isinstance(max_linear_refinements, bool)
            or not isinstance(max_linear_refinements, int)
            or max_linear_refinements not in (0, 1, 2)
        ):
            raise ValueError("CTBR polishing linear refinement budget must be 0, 1, or 2")
        self.max_iter, self.tolerance = max_iter, float(tolerance)
        self.max_linear_refinements = max_linear_refinements
        if (
            isinstance(max_adjoint_refinements, bool)
            or not isinstance(max_adjoint_refinements, int)
            or max_adjoint_refinements not in range(5)
        ):
            raise ValueError("CTBR adjoint refinement budget must be an integer from 0 to 4")
        self.max_adjoint_refinements = max_adjoint_refinements
        self.last_adjoint_diagnostics = None

    @torch.no_grad()
    def solve_adjoint(self, adjoint, qp, hessian, active, upstream):
        """Return v/w/wg only after original-system relative residual checks.

        The original per-row upstream infinity norm is fixed for the initial
        solve and every correction (zero norm uses scale 1). No row transform,
        regularizer, Hessian/active-set update, or gradient sanitation occurs.
        Previously certified rows receive zero correction RHS and retain their
        complete direction while other rows use the remaining bounded budget.
        """
        self.last_adjoint_diagnostics = None
        if upstream.ndim != 3 or not upstream.shape[0] or upstream.dtype != torch.float64:
            raise ValueError("CTBR adjoint requires nonempty float64 upstream[B,T,14]")
        batch, stages, width = upstream.shape
        if width != 14 or qp.A0.shape[-2] != 10 or qp.A_minus.shape[-2] != 10:
            raise ValueError("CTBR adjoint requires the canonical 10-state/4-control layout")
        if active.dtype != torch.bool or active.shape != qp.G.shape[:-1]:
            raise ValueError("CTBR adjoint active mask does not match the original inequalities")

        magnitudes = upstream.abs().flatten(1).amax(-1)
        scales = torch.where(magnitudes == 0, torch.ones_like(magnitudes), magnitudes)
        history_shape = (batch, self.max_adjoint_refinements + 1, 3)
        diagnostics = {
            "scope": "Last adjoint call only; detached and not bound to MPCResult history",
            "tolerance": self.adjoint_tolerance,
            "max_refinements": self.max_adjoint_refinements,
            "upstream_inf_norm": magnitudes.detach().clone(),
            "scales": scales.detach().clone(),
            "normalized_residual_history": upstream.new_full(history_shape, torch.inf),
            "actual_residual_history": upstream.new_full(history_shape, torch.inf),
            "scaled_actual_residual_history": upstream.new_full(history_shape, torch.inf),
            "correction_counts": torch.zeros(batch, dtype=torch.int32, device=upstream.device),
            "batched_correction_calls": 0,
            "passed_rows": torch.zeros(batch, dtype=torch.bool, device=upstream.device),
            "passed": False,
            "error": None,
        }
        self.last_adjoint_diagnostics = diagnostics

        def reject(message):
            diagnostics["error"] = message
            raise RuntimeError(f"CTBR adjoint failed: {message}")

        def row_scale(value):
            return scales.reshape((batch,) + (1,) * (value.ndim - 1))

        def select(mask, selected, previous):
            return torch.where(mask.reshape((batch,) + (1,) * (selected.ndim - 1)), selected, previous)

        original_parts = (upstream, hessian, qp.E, qp.A0, qp.A_minus, qp.A_plus, qp.G)
        if not bool(torch.stack([torch.isfinite(part).all() for part in original_parts]).all()):
            reject("nonfinite original matrix or upstream")

        unit_upstream = upstream / row_scale(upstream)
        equality_rhs = upstream.new_zeros((batch, stages * 10))
        inequality_rhs = upstream.new_zeros(active.shape)
        try:
            unit_solution = adjoint.solve(qp, hessian, active, unit_upstream)
        except RuntimeError as error:
            diagnostics["error"] = f"initial native solve: {error}"
            raise RuntimeError(f"CTBR adjoint failed: {diagnostics['error']}") from error
        # Multiplication owns fresh storage even if a backend returns workspace views.
        v, w, wg = (value * row_scale(value) for value in unit_solution)

        def measure(index):
            actual_defect = _linear_defect(qp, hessian, active, v, w, wg, upstream, equality_rhs, inequality_rhs)
            normalized_defect = _linear_defect(
                qp,
                hessian,
                active,
                v / row_scale(v),
                w / row_scale(w),
                wg / row_scale(wg),
                unit_upstream,
                equality_rhs,
                inequality_rhs,
            )
            actual_error = torch.stack(tuple(_maximum(part) for part in actual_defect), -1)
            normalized_error = torch.stack(tuple(_maximum(part) for part in normalized_defect), -1)
            relative_error = actual_error / scales[:, None]
            finite_rows = torch.stack([torch.isfinite(part).flatten(1).all(-1) for part in (v, w, wg)], -1).all(-1)
            finite_rows &= torch.isfinite(normalized_error).all(-1) & torch.isfinite(relative_error).all(-1)
            passed = finite_rows & (normalized_error <= self.adjoint_tolerance).all(-1)
            passed &= (relative_error <= self.adjoint_tolerance).all(-1)
            diagnostics["normalized_residual_history"][:, index] = normalized_error
            diagnostics["actual_residual_history"][:, index] = actual_error
            diagnostics["scaled_actual_residual_history"][:, index] = relative_error
            diagnostics["actual_residuals"] = actual_error.detach().clone()
            diagnostics["scaled_actual_residuals"] = relative_error.detach().clone()
            diagnostics["normalized_residuals"] = normalized_error.detach().clone()
            diagnostics["actual_defect"] = tuple(part.detach().clone() for part in actual_defect)
            diagnostics["finite_rows"] = finite_rows.detach().clone()
            diagnostics["passed_rows"] = passed.detach().clone()
            if not bool(finite_rows.all()):
                reject("nonfinite adjoint direction or original-system residual")
            return actual_defect, passed

        defect, passed = measure(0)
        for refinement in range(self.max_adjoint_refinements):
            if bool(passed.all()):
                break
            pending = ~passed
            correction_rhs = tuple(select(pending, part / row_scale(part), torch.zeros_like(part)) for part in defect)
            diagnostics["correction_counts"] += pending.to(torch.int32)
            diagnostics["batched_correction_calls"] += 1
            try:
                cv, cw, cwg = adjoint.solve_full(qp, hessian, active, *correction_rhs)
            except RuntimeError as error:
                diagnostics["error"] = f"native correction {refinement + 1}: {error}"
                raise RuntimeError(f"CTBR adjoint failed: {diagnostics['error']}") from error
            v = select(pending, v + cv * row_scale(cv), v)
            w = select(pending, w + cw * row_scale(cw), w)
            wg = select(pending, wg + cwg * row_scale(cwg), wg)
            defect, passed = measure(refinement + 1)

        if not bool(passed.all()):
            failed = (~passed).nonzero().flatten().cpu().tolist()
            reject(f"original-system residual exceeded 1e-11 after bounded refinement; rows={failed}")
        diagnostics["passed"] = True
        return v, w, wg

    def validate_inputs(self, inputs, dynamics):
        """Call BEFORE detaching inputs to enforce the cost-only contract."""
        if type(dynamics) is not CTBRDynamics:
            raise TypeError("CTBR polishing requires the canonical CTBRDynamics model")
        for name in ("initial_state", "G", "lower", "upper"):
            if getattr(inputs, name).requires_grad:
                raise ValueError(f"Cost-only CTBR polishing rejects learnable {name}")
        if inputs.q.ndim != 3 or inputs.q.shape[-1] != 14 or inputs.q.shape[1] < 2:
            raise ValueError("CTBR polishing expects q[B,H+1,14], H>=1")
        b, t, n = inputs.q.shape
        if (
            b < 1
            or inputs.initial_state.shape != (b, 10)
            or inputs.D.shape != (b, t, n, n)
            or inputs.E.shape != (b, t - 1, n, n)
            or inputs.G.ndim != 4
            or inputs.G.shape[:2] != (b, t)
            or inputs.G.shape[-1] != n
            or inputs.lower.shape != inputs.G.shape[:-1]
            or inputs.upper.shape != inputs.lower.shape
            or not inputs.G.shape[-2]
        ):
            raise ValueError("CTBR polishing input dimensions do not match the 10-state/4-input problem")
        if any(
            getattr(inputs, f.name).dtype != torch.float64 or getattr(inputs, f.name).device != inputs.q.device
            for f in fields(inputs)
        ):
            raise ValueError("CTBR polishing inputs must share float64 dtype and device")

    def _active_basis(self, solver, inputs, qp, z, lambda0, lambda_dyn, y):
        value, lower, upper = _physical_constraints(solver, inputs, z)
        dl, du = value - lower, upper - value
        relative = solver.admm.config.eps_rel
        threshold = max(solver.config.tolerance, solver.admm.config.eps_abs)
        lower_scale = torch.where(torch.isfinite(lower), lower.abs(), 0.0)
        upper_scale = torch.where(torch.isfinite(upper), upper.abs(), 0.0)
        near_lower = torch.isfinite(lower) & (dl <= threshold + relative * lower_scale)
        near_upper = torch.isfinite(upper) & (du <= threshold + relative * upper_scale)
        prefer_upper = (y > threshold) | ((y.abs() <= threshold) & (du < dl))
        use_upper = near_upper & (~near_lower | prefer_upper)
        active = (near_lower | near_upper) & (qp.G.abs().amax(-1) > 0)
        bound = torch.where(use_upper, upper, lower)

        # For canonical CTBR: p1=p0+dt*v0 independently of every control.
        # J_plane = n*J_initial_p + dt*n*J_initial_v - n*J_dyn0_p.
        # Preserve the dual contribution with this exact affine identity, then
        # use zero as the explicit gauge for the redundant face multiplier.
        linear_rows = inputs.G.shape[-2]
        row = inputs.G[:, 1]
        prefix = (row[..., :3].abs().amax(-1) > 0) & (row[..., 3:].abs().amax(-1) == 0)
        real_p1 = inputs.initial_state[:, :3] + solver.dynamics.dt * inputs.initial_state[:, 7:10]
        real_value = (row[..., :3] * real_p1[:, None]).sum(-1)
        prefix_violation = _violation(real_value, inputs.lower[:, 1], inputs.upper[:, 1])
        certificate = torch.where(prefix, prefix_violation, 0.0).amax(-1) <= self.tolerance
        weights = torch.where(prefix, y[:, 1, :linear_rows], 0.0)
        transfer = (weights[..., None] * row[..., :3]).sum(-2)
        lambda0[:, :3] += transfer
        lambda0[:, 7:10] += solver.dynamics.dt * transfer
        lambda_dyn[:, 0, :3] -= transfer
        removed = (active[:, 1, :linear_rows] & prefix).sum(-1).to(torch.int32)
        active[:, 1, :linear_rows] &= ~prefix
        y = torch.where(active, y, 0.0)
        return active, bound, lambda0, lambda_dyn, y, certificate, removed

    def __call__(self, solver, inputs, z, admm, accepted_mask):
        self.validate_inputs(inputs, solver.dynamics)
        if z.shape != inputs.q.shape or z.dtype != torch.float64 or z.device != inputs.q.device:
            raise ValueError("CTBR polishing trajectory must match q")
        b, t, _ = z.shape
        m = inputs.G.shape[-2] + (solver.constraints.lower.numel() if solver.constraints is not None else 0)
        if (
            accepted_mask.shape != (b,)
            or accepted_mask.dtype != torch.bool
            or accepted_mask.device != z.device
            or admm.lambda0.shape != (b, 10)
            or admm.lambda_dyn.shape != (b, t - 1, 10)
            or admm.y.shape != (b, t, m)
            or admm.status.shape != (b,)
            or any(x.dtype != z.dtype or x.device != z.device for x in (admm.lambda0, admm.lambda_dyn, admm.y))
        ):
            raise ValueError("CTBR polishing mask or raw multipliers have incompatible shapes")
        with torch.inference_mode(False), torch.no_grad():
            result = PolishedSolution(
                trajectory=z.clone(),
                lambda0=admm.lambda0.clone(),
                lambda_dyn=admm.lambda_dyn.clone(),
                y=admm.y.clone(),
                active_mask=torch.zeros_like(admm.y, dtype=torch.bool),
                status=torch.ones(b, dtype=torch.int32, device=z.device),
                iterations=torch.zeros(b, dtype=torch.int32, device=z.device),
                errors=z.new_full((b, len(POLISH_ERROR_NAMES)), torch.inf),
                linear_residuals=z.new_full((b, self.max_iter, 3), torch.inf),
                failure_reason=torch.ones(b, dtype=torch.int32, device=z.device),
                prefix_rows_removed=torch.zeros(b, dtype=torch.int32, device=z.device),
                linear_residual_history=z.new_full((b, self.max_iter, self.max_linear_refinements + 1, 3), torch.inf),
                linear_refinement_steps=torch.zeros((b, self.max_iter), dtype=torch.int32, device=z.device),
            )
            eligible = accepted_mask & (admm.status == 0)
            for row in range(b):
                if not bool(eligible[row]):
                    continue
                result.status[row] = 2
                result.failure_reason[row] = 2
                data = replace(
                    inputs, **{f.name: getattr(inputs, f.name)[row : row + 1].detach() for f in fields(inputs)}
                )
                current = z[row : row + 1].detach().clone()
                l0 = admm.lambda0[row : row + 1].detach().clone()
                ld = admm.lambda_dyn[row : row + 1].detach().clone()
                y = admm.y[row : row + 1].detach().clone()
                finite = all(
                    torch.isfinite(x).all()
                    for x in (current, l0, ld, y, data.initial_state, data.D, data.E, data.q, data.G)
                )
                valid_bounds = not (
                    torch.isnan(data.lower).any() or torch.isnan(data.upper).any() or (data.lower > data.upper).any()
                )
                if not (finite and valid_bounds):
                    continue
                try:
                    qp = solver._qp(data, current)
                    active, bound, l0, ld, y, prefix_ok, removed = self._active_basis(
                        solver, data, qp, current, l0, ld, y
                    )
                    result.prefix_rows_removed[row] = removed[0]
                    errors = _errors(solver, data, qp, current, l0, ld, y)
                    result.errors[row] = errors[0]
                    if not bool(prefix_ok.all()):
                        result.failure_reason[row] = 3
                        continue
                    valid_step = True
                    result.failure_reason[row] = 6
                    for iteration in range(self.max_iter):
                        if bool(torch.isfinite(errors).all() and (errors <= self.tolerance).all()):
                            break
                        qp = solver._qp(data, current)
                        hessian = qp.D.clone()
                        hessian[:, :-1] += solver._dynamics_hessian(current[:, :-1], ld)
                        if solver.constraints is not None:
                            hessian += solver._constraint_hessian(current, y[..., data.G.shape[-2] :])
                        value, _, _ = _physical_constraints(solver, data, current)
                        primal_rhs = -_stationarity(data, qp, current, l0, ld, y)
                        equality_rhs = -_equalities(solver, data, current)
                        inequality_rhs = torch.where(active, bound - value, 0.0)
                        result.iterations[row] += 1
                        adjoint = solver._adjoint(qp)
                        dz, dlambda, dy = adjoint.solve_full(
                            qp, hessian, active, primal_rhs, equality_rhs, inequality_rhs
                        )
                        # Own the accumulated direction even if a native
                        # workspace reuses its output buffer on the next call.
                        dz, dlambda, dy = dz.clone(), dlambda.clone(), dy.clone()
                        defect = _linear_defect(
                            qp, hessian, active, dz, dlambda, dy, primal_rhs, equality_rhs, inequality_rhs
                        )
                        linear_error = torch.stack(tuple(_maximum(part) for part in defect), -1)
                        result.linear_residuals[row, iteration] = linear_error[0]
                        result.linear_residual_history[row, iteration, 0] = linear_error[0]
                        for refinement in range(self.max_linear_refinements):
                            if not bool(torch.isfinite(linear_error).all()) or bool(
                                (linear_error <= self.tolerance).all()
                            ):
                                break
                            result.linear_refinement_steps[row, iteration] += 1
                            cz, clambda, cy = adjoint.solve_full(qp, hessian, active, *defect)
                            dz, dlambda, dy = dz + cz, dlambda + clambda, dy + cy
                            # Always test the accumulated direction against the
                            # original RHS, never only the correction equation.
                            defect = _linear_defect(
                                qp, hessian, active, dz, dlambda, dy, primal_rhs, equality_rhs, inequality_rhs
                            )
                            linear_error = torch.stack(tuple(_maximum(part) for part in defect), -1)
                            result.linear_residuals[row, iteration] = linear_error[0]
                            result.linear_residual_history[row, iteration, refinement + 1] = linear_error[0]
                        if not bool(torch.isfinite(linear_error).all() and (linear_error <= self.tolerance).all()):
                            valid_step = False
                            result.failure_reason[row] = 5
                            break
                        current = current + dz
                        l0 = l0 + dlambda[:, :10]
                        ld = ld + dlambda[:, 10:].reshape_as(ld)
                        y = torch.where(active, y + dy, 0.0)
                        qp = solver._qp(data, current)
                        errors = _errors(solver, data, qp, current, l0, ld, y)
                        result.errors[row] = errors[0]
                    if not (valid_step and bool(torch.isfinite(errors).all() and (errors <= self.tolerance).all())):
                        continue
                    result.trajectory[row] = current[0]
                    result.lambda0[row], result.lambda_dyn[row], result.y[row] = l0[0], ld[0], y[0]
                    result.active_mask[row] = active[0]
                    result.status[row] = 0
                    result.failure_reason[row] = 0
                except RuntimeError:
                    # cuDSS failures (including unsupported generic dependence)
                    # stay local. Original primal/dual output rows are unchanged.
                    result.failure_reason[row] = 4
                    continue
            return result
