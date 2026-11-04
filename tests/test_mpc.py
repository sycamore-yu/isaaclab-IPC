import numpy as np
import osqp
import pytest
import torch
from scipy import sparse

from isaaclab_ipc.frontend import PlanningContext
from isaaclab_ipc.mpc import MPC, Config
from isaaclab_ipc.turbompc.ipc import FixedIPCSolver


def oracle(problem):
    h, g, a, lower, upper, _ = problem
    solver = osqp.OSQP()
    solver.setup(P=sparse.csc_matrix(np.triu(h)), q=g, A=sparse.csc_matrix(a),
                 l=lower, u=upper, eps_abs=1e-8, eps_rel=1e-8, max_iter=20000,
                 verbose=False, polishing=True)
    result = solver.solve(raise_error=False)
    assert result.info.status == "solved"
    return result.x


def test_prediction_and_release_hessian():
    m = MPC(Config.load())
    x = np.arange(9) * 0.01
    u = np.linspace(-1, 1, 45)
    history = []
    cur = x.copy()
    for j in u.reshape(-1, 3):
        cur = m.a @ cur + m.b @ j
        history.extend(cur)
    np.testing.assert_allclose(m.M @ x + m.C @ u, history, atol=1e-12)
    residual = m.H - m.C.T @ (m.q[:, None] * m.C)
    np.testing.assert_allclose(residual, np.diag(np.repeat([0.2] + [0.4] * 13 + [0.2], 3)), atol=1e-12)
    assert np.linalg.eigvalsh(m.H).min() > 0


def test_hover_and_goal_qp():
    if not torch.cuda.is_available():
        pytest.skip("Native TurboMPC requires CUDA")
    m = MPC(Config.load())
    x = np.zeros(9)
    x[2] = 1
    ref = np.tile(x, (15, 1))
    faces = [np.empty((0, 4)) for _ in range(15)]
    with FixedIPCSolver(device="cuda:0", compile_kernels=False) as solver:
        hover = solver.solve([PlanningContext(x, ref, faces, True)])[0]
        assert hover["status"] == "solved"
        np.testing.assert_allclose(hover["solution"], 0, atol=1e-6)
        ref[:, 0] = np.linspace(0, 2, 15)
        result = solver.solve([PlanningContext(x, ref, faces, True)])[0]
    assert result["status"] == "solved"
    u = result["solution"]
    p = m.problem(x, ref, faces)
    assert np.max(np.maximum(p[2] @ u - p[4], p[3] - p[2] @ u)) < 0.02
    assert u[0] > 0


def test_nonfinite_state_rejected():
    import pytest

    m = MPC(Config.load())
    with pytest.raises(ValueError):
        m.problem(np.full(9, np.nan), np.zeros((15, 9)), [[]] * 15)


def test_fixed_native_solver_matches_osqp():
    if not torch.cuda.is_available():
        pytest.skip("Native TurboMPC requires CUDA")

    m = MPC(Config.load())
    x = np.zeros(9)
    x[2] = 1.0
    ref = np.tile(x, (15, 1))
    ref[:, 0] = np.linspace(0, 2, 15)
    problem = m.problem(x, ref, [[]] * 15)
    expected = oracle(problem)
    context = PlanningContext(x, ref, [np.empty((0, 4)) for _ in range(15)], True)
    with FixedIPCSolver(device="cuda:0", compile_kernels=False) as solver:
        rows = solver.solve([context, context])
    assert [row["status"] for row in rows] == ["solved", "solved"]
    np.testing.assert_allclose(rows[0]["solution"][:3], expected[:3], rtol=0.02, atol=0.03)
