"""Generic CTBR shapes and a known feasible low-cost/high-rho Schur system."""

import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from isaaclab_ipc.turbompc import ADMMConfig, ADMMResult, NativeADMM, SQPConfig, TorchMPC
from isaaclab_ipc.turbompc.ctbr import CTBRConfig, CTBRQPBuilder
from isaaclab_ipc.turbompc.native import schur_matrix


def ctbr_problem(nodes, *, low_cost=False):
    builder = CTBRQPBuilder(CTBRConfig(cost_nodes=nodes)).to(device="cuda", dtype=torch.float64)
    state = torch.zeros((2, 10), device="cuda", dtype=torch.float64)
    state[:, 3] = 1.0
    state[1, :3] = state.new_tensor([0.2, -0.1, 0.4])
    state[1, 7:10] = state.new_tensor([0.1, -0.05, 0.02])
    control = builder.hover_control.expand(2, nodes, -1).clone()
    control[1, :, 1:] = state.new_tensor([0.1, -0.08, 0.05])
    reference = builder.dynamics().rollout(state, control[:, :-1])
    if not low_cost:
        reference = reference.clone()
        reference[:, :, 0] += torch.linspace(0.0, 0.1, nodes, device=state.device)
    planes = state.new_zeros((2, nodes, 2, 4))
    planes[:, :, 0, 0], planes[:, :, 1, 0] = 1.0, -1.0
    planes[..., 3] = -5.0
    observation = dict(
        ctbr_state=state,
        ctbr_reference=reference,
        ctbr_control_reference=control,
        ctbr_warm_start=control[:, :-1],
        planes=planes,
        plane_mask=torch.ones((2, nodes, 2), dtype=torch.bool, device=state.device),
    )
    costs = state.new_zeros((2, builder.config.cost_dim))
    if low_cost:
        costs[:, : 14 * nodes] = -80.0
    else:
        costs[1, : 14 * nodes] = 0.2
    inputs, guess = builder(observation, costs)
    return builder, inputs, guess


@pytest.mark.parametrize("nodes", [2, 5, 10])
def test_inverse_ctbr_generic_node_count_preserves_solution(nodes):
    builder, inputs, guess = ctbr_problem(nodes)
    results = []
    for mode in ("cudss_vector", "cudss_inverse_chunk"):
        solver = TorchMPC(
            builder.dynamics(),
            SQPConfig(max_iter=25, tolerance=1e-4, linesearch="none"),
            ADMMConfig(linear_solver=mode),
            constraints=builder.path_constraints(),
        )
        try:
            result = solver.solve(inputs, guess)
            assert (result.status == 0).all(), (mode, result.status, result.errors)
            assert (result.errors <= 1e-4).all(), result.errors
            assert (result.normal_cone_error <= 1e-4).all(), result.normal_cone_error
            assert (result.controls >= builder.control_lower - 1e-4).all()
            assert (result.controls <= builder.control_upper + 1e-4).all()
            results.append(result)
        finally:
            solver.close()
    torch.testing.assert_close(results[1].trajectory, results[0].trajectory, atol=1e-4, rtol=0)


def dense_schur(matrix, row):
    """Assemble the actual uploaded banded S without reordering its sums."""
    banded = matrix[row].cpu().numpy()
    nodes, width, _ = banded.shape
    dense = np.zeros((nodes * width, nodes * width), dtype=np.float64)
    for stage in range(nodes):
        block = slice(stage * width, (stage + 1) * width)
        dense[block, block] = banded[stage, :, width : 2 * width]
        if stage:
            dense[block, slice((stage - 1) * width, stage * width)] = banded[stage, :, :width]
        if stage + 1 < nodes:
            dense[block, slice((stage + 1) * width, (stage + 2) * width)] = banded[stage, :, 2 * width :]
    return dense


def dense_constraints(qp, row):
    nodes, width = qp.q.shape[1:]
    nx, n0, inequalities = qp.A_minus.shape[-2], qp.A0.shape[-2], qp.G.shape[-2]
    equality = np.zeros((n0 + (nodes - 1) * nx, nodes * width))
    bounds = np.zeros((nodes * inequalities, nodes * width))
    equality[:n0, :width] = qp.A0[row].cpu().numpy()
    for stage in range(nodes):
        block = slice(stage * width, (stage + 1) * width)
        bounds[stage * inequalities : (stage + 1) * inequalities, block] = qp.G[row, stage].cpu().numpy()
        if stage + 1 < nodes:
            eq = slice(n0 + stage * nx, n0 + (stage + 1) * nx)
            equality[eq, block] = qp.A_minus[row, stage].cpu().numpy()
            equality[eq, slice((stage + 1) * width, (stage + 2) * width)] = qp.A_plus[row, stage].cpu().numpy()
    target = np.r_[qp.c0[row].cpu().numpy(), qp.c_dyn[row].cpu().numpy().ravel()]
    return equality, bounds, target


def test_inverse_actual_ctbr_low_diagonal_high_rho(record_property):
    builder, inputs, guess = ctbr_problem(5, low_cost=True)
    # A rollout with the same reference/control cost has a known feasible
    # optimum. Its linearized C and G are the production CTBR constraints.
    mpc = TorchMPC(builder.dynamics(), constraints=builder.path_constraints())
    try:
        qp = mpc._qp(inputs, guess)
    finally:
        mpc.close()
    assert qp.D.diagonal(dim1=-2, dim2=-1).amin().item() == pytest.approx(2e-5)
    assert torch.count_nonzero(qp.E) == 0
    batch = len(guess)
    high_rho = 1e6
    warm = ADMMResult(
        iterations=torch.zeros(batch, dtype=torch.int32, device=guess.device),
        x=guess.clone(),
        z=(qp.G @ guess.unsqueeze(-1)).squeeze(-1),
        y=torch.zeros_like(qp.lower),
        lambda0=torch.zeros_like(qp.c0),
        lambda_dyn=torch.zeros_like(qp.c_dyn),
        xi=torch.zeros_like(qp.lower),
        rho=guess.new_full((batch,), high_rho),
        kernel_ns=guess.new_zeros(batch),
        status=torch.full((batch,), -1, dtype=torch.int32, device=guess.device),
        residuals=guess.new_zeros((batch, 4)),
    )
    config = ADMMConfig(rho=high_rho, max_iter=1, adapt_rho_every=0)
    matrix = schur_matrix(qp, warm.rho, config)
    systems = []
    for row in range(batch):
        dense = dense_schur(matrix, row)
        equality, bounds, target = dense_constraints(qp, row)
        original = guess[row].cpu().numpy().ravel()
        np.testing.assert_allclose(equality @ original, target, atol=1e-12, rtol=0)
        assert np.linalg.matrix_rank(np.r_[equality, bounds]) == len(original)
        gamma = (
            config.sigma * original - qp.q[row].cpu().numpy().ravel()
            + config.rho_f_factor * high_rho * equality.T @ target
            + high_rho * bounds.T @ warm.z[row].cpu().numpy().ravel()
        )
        eigenvalues = np.linalg.eigvalsh(dense)
        assert eigenvalues[0] > 0
        systems.append((dense, gamma, np.linalg.solve(dense, gamma), original, eigenvalues))
    evidence = {"D_min": float(qp.D.diagonal(dim1=-2, dim2=-1).amin()), "rho": high_rho, "modes": {}}
    completed = []
    for mode in ("cudss_vector", "cudss_inverse_chunk"):
        solver = NativeADMM(replace(config, linear_solver=mode))
        try:
            first = solver.solve(qp, warm)
        finally:
            solver.close()
        rows = []
        for row, (dense, gamma, reference, original, eigenvalues) in enumerate(systems):
            # Undo only the documented primal over-relaxation of this one step.
            actual = (first.x[row].cpu().numpy().ravel() - (1 - config.alpha) * original) / config.alpha
            residual = np.linalg.norm(dense @ actual - gamma, ord=np.inf)
            scale = np.linalg.norm(dense, ord=np.inf) * np.linalg.norm(actual, ord=np.inf) + np.linalg.norm(gamma, ord=np.inf)
            normalized = residual / scale
            assert normalized <= 1e-11, (mode, row, normalized)
            np.testing.assert_allclose(actual, reference, atol=2e-7, rtol=1e-7)
            rows.append(dict(
                lambda_min=float(eigenvalues[0]), lambda_max=float(eigenvalues[-1]),
                condition_number=float(eigenvalues[-1] / eigenvalues[0]),
                linear_residual_inf=float(residual), normalized_linear_residual=float(normalized),
                raw_status=int(first.status[row]), raw_iterations=int(first.iterations[row]),
                raw_residuals=first.residuals[row].cpu().tolist(),
            ))
        # This is a convergence gate with the unchanged production budget and
        # rho checks, separate from the deliberately truncated linear step.
        solver = NativeADMM(ADMMConfig(rho=high_rho, linear_solver=mode))
        try:
            result = solver.solve(qp, warm)
            assert (result.status == 0).all(), (mode, result.status, result.residuals)
            assert (result.residuals[:, :2] <= result.residuals[:, 2:]).all(), result.residuals
            torch.testing.assert_close(result.x, guess, atol=1e-4, rtol=0)
            completed.append(result)
            evidence["modes"][mode] = dict(
                first_linear_step=rows, status=result.status.cpu().tolist(),
                iterations=result.iterations.cpu().tolist(), residuals=result.residuals.cpu().tolist(),
                final_rho=result.rho.cpu().tolist(),
            )
        finally:
            solver.close()
    torch.testing.assert_close(completed[1].x, completed[0].x, atol=1e-4, rtol=0)
    record_property("ctbr_low_diagonal_high_rho", json.dumps(evidence))
