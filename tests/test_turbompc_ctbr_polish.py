"""CPU reference and failure-isolation checks for the device CTBR polisher.

The small fixture is a saved native CTBR plan at the predeclared wall case's
index7. Reference multipliers come from an independent dense KKT calculation,
not from unsaved CUDA duals. The production module never imports this CPU
reference or its NumPy fixture loader.
"""

import hashlib
import json
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest
import torch

from isaaclab_ipc.turbompc import ADMMConfig, LinearDynamics, MPCInputs, SQPConfig, TorchMPC
from isaaclab_ipc.turbompc.ctbr import CTBRQPBuilder
from isaaclab_ipc.turbompc.ctbr_polish import POLISH_ERROR_NAMES, CTBRPolisher
from isaaclab_ipc.turbompc.native import ADMMResult

FIXTURE = Path(__file__).with_name("data") / "ctbr_polish_reference.npz"
FIXTURE_SHA = "3a983a4963e8d0f288c46c40dc07ec4579ba136c7d394d5c762ba0d30b66ecde"


@pytest.fixture(scope="module", autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class DenseReference:
    """Independent full KKT matrix construction from the public QP metadata."""

    def __init__(self, *, corrupt=False, corrupt_calls=(), raise_error=False, reuse_buffer=False):
        self.calls = 0
        self.hessians = []
        self.off_diagonals = []
        self.systems = []
        self.corrupt_calls, self.reuse_buffer = corrupt_calls, reuse_buffer
        self.buffer = None
        self.corrupt, self.raise_error = corrupt, raise_error

    def solve_full(self, qp, hessian, active, primal_rhs, equality_rhs, inequality_rhs):
        self.calls += 1
        self.hessians.append(hessian.clone())
        self.off_diagonals.append(qp.E.clone())
        if self.raise_error:
            raise RuntimeError("Reference factorization failure")
        b, t, n = primal_rhs.shape
        nx, m = qp.A0.shape[1], qp.G.shape[-2]
        assert b == 1
        primal, eq = t * n, t * nx
        h = torch.block_diag(*hessian[0])
        c = h.new_zeros((eq, primal))
        c[:nx, :n] = qp.A0[0]
        for k in range(t - 1):
            h[(k + 1) * n : (k + 2) * n, k * n : (k + 1) * n] = qp.E[0, k]
            h[k * n : (k + 1) * n, (k + 1) * n : (k + 2) * n] = qp.E[0, k].T
            c[(k + 1) * nx : (k + 2) * nx, k * n : (k + 1) * n] = qp.A_minus[0, k]
            c[(k + 1) * nx : (k + 2) * nx, (k + 1) * n : (k + 2) * n] = qp.A_plus[0, k]
        g = torch.block_diag(*qp.G[0]) * active[0].flatten()[:, None]
        matrix = torch.cat(
            (
                torch.cat((h, c.T, g.T), 1),
                torch.cat((c, h.new_zeros(eq, eq + t * m)), 1),
                torch.cat((g, h.new_zeros(t * m, eq), torch.diag((~active[0]).flatten().double())), 1),
            ),
            0,
        )
        rhs = torch.cat((primal_rhs.flatten(), equality_rhs.flatten(), inequality_rhs.flatten()))
        solution = torch.linalg.solve(matrix, rhs)
        if self.corrupt or self.calls in self.corrupt_calls:
            # Mimic a finite native solve whose equality block was inaccurate.
            # Checking stationarity alone is not an acceptable certificate.
            solution[0] += 1e-7
        self.systems.append((qp, hessian, active, matrix.clone(), rhs.clone(), solution.clone()))
        if self.reuse_buffer:
            if self.buffer is None:
                self.buffer = solution.clone()
            else:
                self.buffer.copy_(solution)
            solution = self.buffer
        return (
            solution[:primal].reshape(b, t, n),
            solution[primal : primal + eq].reshape(b, eq),
            solution[primal + eq :].reshape(b, t, m),
        )


def raw_result(z, lambda0, lambda_dyn, y):
    batch = len(z)
    return ADMMResult(
        torch.ones(batch, dtype=torch.int32),
        z.clone(),
        torch.zeros_like(y),
        y,
        lambda0,
        lambda_dyn,
        torch.zeros_like(y),
        z.new_full((batch,), 0.1),
        torch.zeros(batch, dtype=torch.int64),
        torch.zeros(batch, dtype=torch.int32),
        z.new_zeros((batch, 4)),
    )


@pytest.fixture
def problem():
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == FIXTURE_SHA
    with np.load(FIXTURE, allow_pickle=False) as data:
        provenance = json.loads(str(data["provenance"]))
        assert provenance["model_index"] == 7
        inputs = MPCInputs(
            **{name: torch.from_numpy(data[name].copy())[None] for name in MPCInputs.__dataclass_fields__}
        )
        z = torch.from_numpy(data["trajectory"].copy())[None]
        admm = raw_result(
            z,
            torch.from_numpy(data["original_lambda0"].copy())[None],
            torch.from_numpy(data["original_lambda_dyn"].copy())[None],
            torch.from_numpy(data["original_y"].copy())[None],
        )
        expected = {
            name: torch.from_numpy(data[name].copy())[None] for name in data.files if name.startswith("reference_")
        }
    builder = CTBRQPBuilder()
    solver = TorchMPC(
        builder.dynamics(),
        SQPConfig(tolerance=1e-7),
        ADMMConfig(eps_abs=1e-8, eps_rel=0),
        constraints=builder.path_constraints(),
    )
    reference = DenseReference()
    solver._adjoint = lambda qp: reference
    yield solver, inputs, z, admm, expected, reference
    solver.close()


def assert_unchanged(result, z, admm, rows=slice(None)):
    for actual, original in (
        (result.trajectory, z),
        (result.lambda0, admm.lambda0),
        (result.lambda_dyn, admm.lambda_dyn),
        (result.y, admm.y),
    ):
        torch.testing.assert_close(actual[rows], original[rows], rtol=0, atol=0, equal_nan=True)
    assert not result.active_mask[rows].any()


def test_original_wall_plan_polishes_with_full_curvature_and_preserves_raw_admm(problem):
    solver, inputs, z, admm, expected, reference = problem
    snapshot = {f.name: getattr(admm, f.name).clone() for f in fields(admm)}
    inputs_before = {f.name: getattr(inputs, f.name).clone() for f in fields(inputs)}
    result = CTBRPolisher()(solver, inputs, z, admm, torch.tensor([True]))
    assert result.status.tolist() == [0]
    assert 1 <= result.iterations.item() <= 4
    assert result.errors.shape == (1, len(POLISH_ERROR_NAMES))
    assert result.errors.max() <= 1e-11
    assert result.failure_reason.tolist() == [0]
    assert result.prefix_rows_removed.tolist() == [1]
    assert result.linear_residuals[0, : result.iterations.item()].max() <= 1e-11
    assert result.linear_residual_history.shape == (1, 4, 3, 3)
    assert result.linear_refinement_steps.shape == (1, 4)
    assert not result.linear_refinement_steps.any()
    torch.testing.assert_close(result.linear_residual_history[:, :, 0], result.linear_residuals, rtol=0, atol=0)
    assert torch.isinf(result.linear_residual_history[:, :, 1:]).all()
    assert not result.active_mask[0, 1, 7]
    assert (reference.hessians[0] - inputs.D).abs().max() > 1e-4
    torch.testing.assert_close(result.trajectory, expected["reference_trajectory"], atol=2e-11, rtol=0)
    torch.testing.assert_close(result.lambda0, expected["reference_lambda0"], atol=2e-8, rtol=0)
    torch.testing.assert_close(result.lambda_dyn, expected["reference_lambda_dyn"], atol=2e-8, rtol=0)
    torch.testing.assert_close(result.y, expected["reference_y"], atol=2e-8, rtol=0)
    assert result.trajectory[0, 0, 10] - z[0, 0, 10] == pytest.approx(1.0405560927573276e-5, abs=2e-11)
    for f in fields(admm):
        torch.testing.assert_close(getattr(admm, f.name), snapshot[f.name], rtol=0, atol=0)
    for f in fields(inputs):
        torch.testing.assert_close(getattr(inputs, f.name), inputs_before[f.name], rtol=0, atol=0)


@pytest.mark.parametrize("name", ["initial_state", "G", "lower", "upper"])
def test_learnable_physical_inputs_rejected_before_detachment(problem, name):
    solver, inputs, *_ = problem
    changed = replace(inputs, **{name: getattr(inputs, name).clone().requires_grad_()})
    with pytest.raises(ValueError, match=f"learnable {name}"):
        CTBRPolisher().validate_inputs(changed, solver.dynamics)


def test_cost_parameters_including_off_diagonal_are_supported(problem):
    solver, inputs, z, admm, _, reference = problem
    # Add a genuine inter-stage term and compensate its gradient at the saved
    # point. The same starting trajectory remains near the same solution, but
    # its cost curvature and implicit refinement are different.
    e = inputs.E.clone()
    e[..., 10, 10] = 1e-3
    q = inputs.q.clone()
    q[:, 1:] -= torch.einsum("btij,btj->bti", e, z[:, :-1])
    q[:, :-1] -= torch.einsum("btji,btj->bti", e, z[:, 1:])
    changed = replace(inputs, D=inputs.D.clone().requires_grad_(), E=e.requires_grad_(), q=q.requires_grad_())
    CTBRPolisher().validate_inputs(changed, solver.dynamics)
    result = CTBRPolisher()(solver, changed, z, admm, torch.tensor([True]))
    assert result.status.tolist() == [0]
    assert result.errors.max() <= 1e-11
    assert reference.calls > 0
    torch.testing.assert_close(reference.off_diagonals[0], e, rtol=0, atol=0)


@pytest.mark.parametrize("outer_accepted,raw_status", [(False, 0), (True, 1)])
def test_unaccepted_rows_skip_kkt_and_keep_original_data(problem, outer_accepted, raw_status):
    solver, inputs, z, admm, _, reference = problem
    admm = replace(admm, status=torch.tensor([raw_status], dtype=torch.int32))
    result = CTBRPolisher()(solver, inputs, z, admm, torch.tensor([outer_accepted]))
    assert result.status.tolist() == [1]
    assert result.iterations.item() == 0
    assert reference.calls == 0
    assert torch.isinf(result.errors).all()
    assert torch.isinf(result.linear_residual_history).all()
    assert not result.linear_refinement_steps.any()
    assert_unchanged(result, z, admm)


@pytest.mark.parametrize("failure", ["inaccurate", "exception"])
def test_failed_native_kkt_cannot_certify_or_replace_a_plan(problem, failure):
    solver, inputs, z, admm, *_ = problem
    native = DenseReference(corrupt=failure == "inaccurate", raise_error=failure == "exception")
    solver._adjoint = lambda qp: native
    result = CTBRPolisher()(solver, inputs, z, admm, torch.tensor([True]))
    assert result.status.tolist() == [2]
    assert result.iterations.item() == 1
    assert result.failure_reason.tolist() == [5 if failure == "inaccurate" else 4]
    if failure == "inaccurate":
        assert result.linear_residuals[0, 0].max() > 1e-11
        assert native.calls == 3
        assert result.linear_refinement_steps[0, 0] == 2
    else:
        assert native.calls == 1
        assert result.linear_refinement_steps[0, 0] == 0
    assert_unchanged(result, z, admm)


def test_fixed_first_position_outside_original_wall_rejects_without_solving(problem):
    solver, inputs, z, admm, _, reference = problem
    state = inputs.initial_state.clone()
    state[0, 0] = 0.2 + 1e-8 - solver.dynamics.dt * state[0, 7]
    changed = replace(inputs, initial_state=state)
    result = CTBRPolisher()(solver, changed, z, admm, torch.tensor([True]))
    assert result.status.tolist() == [2]
    assert result.iterations.item() == 0
    assert result.failure_reason.tolist() == [3]
    assert reference.calls == 0
    assert_unchanged(result, z, admm)


def test_mixed_rows_isolate_nonfinite_and_unattempted_samples(problem):
    solver, inputs, z, admm, _, reference = problem
    data = replace(
        inputs,
        **{
            f.name: getattr(inputs, f.name).repeat(3, *([1] * (getattr(inputs, f.name).ndim - 1)))
            for f in fields(inputs)
        },
    )
    trajectory = z.repeat(3, 1, 1)
    trajectory[1, 0, 0] = float("nan")
    raw = ADMMResult(
        **{f.name: getattr(admm, f.name).repeat(3, *([1] * (getattr(admm, f.name).ndim - 1))) for f in fields(admm)}
    )
    result = CTBRPolisher()(solver, data, trajectory, raw, torch.tensor([True, True, False]))
    assert result.status.tolist() == [0, 2, 1]
    assert reference.calls == result.iterations[0].item()
    assert_unchanged(result, trajectory, raw, slice(1, None))


def test_duplicate_generic_face_is_not_silently_removed(problem):
    solver, inputs, z, admm, *_ = problem
    changed = replace(
        inputs,
        G=torch.cat((inputs.G, inputs.G[:, :, 7:8]), -2),
        lower=torch.cat((inputs.lower, inputs.lower[:, :, 7:8]), -1),
        upper=torch.cat((inputs.upper, inputs.upper[:, :, 7:8]), -1),
    )
    y = torch.cat((admm.y[..., :13], torch.zeros_like(admm.y[..., :1]), admm.y[..., 13:]), -1)
    raw = replace(admm, y=y, z=torch.zeros_like(y), xi=torch.zeros_like(y))
    result = CTBRPolisher()(solver, changed, z, raw, torch.tensor([True]))
    assert result.status.tolist() == [2]
    assert_unchanged(result, z, raw)


def test_insufficient_fixed_newton_budget_returns_original_data(problem):
    solver, inputs, z, admm, *_ = problem
    result = CTBRPolisher(max_iter=1)(solver, inputs, z, admm, torch.tensor([True]))
    assert result.status.tolist() == [2]
    assert result.iterations.tolist() == [1]
    assert result.failure_reason.tolist() == [6]
    assert result.errors.max() > 1e-11
    assert result.linear_residuals.max() <= 1e-11
    assert_unchanged(result, z, admm)


def test_already_certified_solution_needs_no_extra_native_solve(problem):
    solver, inputs, _, _, expected, reference = problem
    z = expected["reference_trajectory"]
    raw = raw_result(z, expected["reference_lambda0"], expected["reference_lambda_dyn"], expected["reference_y"])
    result = CTBRPolisher()(solver, inputs, z, raw, torch.tensor([True]))
    assert result.status.tolist() == [0]
    assert result.iterations.tolist() == [0]
    assert result.errors.max() <= 1e-11
    assert reference.calls == 0


def test_wrong_dynamics_cannot_use_the_ctbr_prefix_certificate(problem):
    _, inputs, *_ = problem
    with pytest.raises(TypeError, match="canonical CTBRDynamics"):
        CTBRPolisher().validate_inputs(inputs, LinearDynamics(torch.eye(10, 14, dtype=torch.float64), nx=10))


@pytest.mark.parametrize("budget", [-1, 3, True, False, 1.0, None])
def test_linear_refinement_budget_must_be_bounded_integer(budget):
    with pytest.raises(ValueError, match="0, 1, or 2"):
        CTBRPolisher(max_linear_refinements=budget)


@pytest.mark.parametrize("budget", [0, 1, 2])
def test_inaccurate_native_solves_have_exact_bounded_correction_count(problem, budget):
    solver, inputs, z, admm, *_ = problem
    native = DenseReference(corrupt=True)
    solver._adjoint = lambda qp: native
    result = CTBRPolisher(max_linear_refinements=budget)(solver, inputs, z, admm, torch.tensor([True]))
    assert result.status.tolist() == [2]
    assert result.failure_reason.tolist() == [5]
    assert result.iterations.tolist() == [1]
    assert native.calls == 1 + budget
    assert result.linear_refinement_steps[0].tolist() == [budget, 0, 0, 0]
    assert result.linear_residual_history.shape == (1, 4, budget + 1, 3)
    history = result.linear_residual_history[0, 0]
    assert torch.isfinite(history).all()
    assert (history.max(-1).values > 1e-11).all()
    torch.testing.assert_close(result.linear_residuals[0, 0], history[-1], rtol=0, atol=0)
    assert torch.isinf(result.linear_residual_history[0, 1:]).all()
    assert_unchanged(result, z, admm)


def test_original_kkt_defect_correction_uses_frozen_matrix_and_owned_direction(problem):
    solver, inputs, z, admm, expected, _ = problem
    # A re-used output buffer ensures the original direction must be retained
    # independently of the correction solve's output storage.
    native = DenseReference(corrupt_calls=(1,), reuse_buffer=True)
    solver._adjoint = lambda qp: native
    raw_before = {f.name: getattr(admm, f.name).clone() for f in fields(admm)}
    result = CTBRPolisher()(solver, inputs, z, admm, torch.tensor([True]))
    assert result.status.tolist() == [0]
    assert result.errors.max() <= 1e-11
    assert result.linear_refinement_steps.sum() == 1
    assert native.calls == result.iterations.item() + 1
    first, correction = native.systems[:2]
    assert all(first[i] is correction[i] for i in range(3))
    torch.testing.assert_close(first[3], correction[3], rtol=0, atol=0)
    expected_defect = first[4] - first[3] @ first[5]
    torch.testing.assert_close(correction[4], expected_defect, rtol=0, atol=2e-13)
    accumulated = first[5] + correction[5]
    assert (first[3] @ accumulated - first[4]).abs().max() <= 1e-11
    history = result.linear_residual_history[0, 0]
    assert history[0].max() > 1e-11
    assert history[1].max() <= 1e-11
    assert torch.isinf(history[2]).all()
    torch.testing.assert_close(result.linear_residuals[0, 0], history[1], rtol=0, atol=0)
    torch.testing.assert_close(result.trajectory, expected["reference_trajectory"], rtol=0, atol=2e-11)
    for f in fields(admm):
        torch.testing.assert_close(getattr(admm, f.name), raw_before[f.name], rtol=0, atol=0)


def test_linear_correction_cannot_replace_the_nonlinear_certificate(problem):
    solver, inputs, z, admm, *_ = problem
    native = DenseReference(corrupt_calls=(1,))
    solver._adjoint = lambda qp: native
    result = CTBRPolisher(max_iter=1)(solver, inputs, z, admm, torch.tensor([True]))
    assert native.calls == 2
    assert result.linear_refinement_steps.tolist() == [[1]]
    assert result.linear_residuals.max() <= 1e-11
    assert result.errors.max() > 1e-11
    assert result.status.tolist() == [2]
    assert result.failure_reason.tolist() == [6]
    assert_unchanged(result, z, admm)
