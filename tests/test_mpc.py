import numpy as np

from isaaclab_ipc.mpc import MPC, Config


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
    m = MPC(Config.load())
    x = np.zeros(9)
    x[2] = 1
    ref = np.tile(x, (15, 1))
    p = m.problem(x, ref, [[]] * 15)
    u, status, pr, du = m.solve(p)
    assert status == "solved"
    np.testing.assert_allclose(u, 0, atol=1e-9)
    ref[:, 0] = np.linspace(0, 2, 15)
    p = m.problem(x, ref, [[]] * 15)
    u, status, pr, du = m.solve(p)
    assert status == "solved"
    assert np.max(np.maximum(p[2] @ u - p[4], p[3] - p[2] @ u)) < 0.02
    assert u[0] > 0


def test_nonfinite_state_rejected():
    import pytest

    m = MPC(Config.load())
    with pytest.raises(ValueError):
        m.problem(np.full(9, np.nan), np.zeros((15, 9)), [[]] * 15)


def test_batch_matches_osqp():
    from isaaclab_ipc.qp import BatchedQP

    m = MPC(Config.load())
    x = np.zeros(9)
    x[2] = 1.0
    ref = np.tile(x, (15, 1))
    ref[:, 0] = np.linspace(0, 2, 15)
    problem = m.problem(x, ref, [[]] * 15)
    expected, status, _, _ = m.solve(problem)
    solver = BatchedQP(2, 45, len(problem[3]), device="cpu", max_iterations=16384, eps_abs=1e-5, eps_rel=0.0)
    values, statuses, pr, du, it = solver.solve([problem, problem])
    assert statuses == ["solved"] * 2, (pr, du, it)
    np.testing.assert_allclose(values[0, :3], expected[:3], rtol=0.02, atol=0.03)
