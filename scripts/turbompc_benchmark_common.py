"""Identical serialized mathematical inputs for native and upstream benchmarks."""

import numpy as np

UPSTREAM_COMMIT = "7050a4f367d08504ecb6f5820d9161e704df5816"


def make_case(batch: int, horizon: int, nonlinear: bool):
    rng = np.random.default_rng(719)
    nx, nu = (2, 1) if nonlinear else (9, 3)
    dt = 0.1
    x0 = rng.uniform(-0.15, 0.15, (batch, nx))
    reference = np.zeros(nx)
    if nonlinear:
        reference[0] = 0.7
        diagonal = np.array([3.0, 0.4, 0.2])
    else:
        reference[:3] = [1.0, -0.6, 0.8]
        diagonal = np.array([3.0] * 3 + [0.4] * 3 + [0.1] * 3 + [0.05] * 3)
    costs = rng.uniform(-0.2, 0.2, (batch, nx + nu))
    probe = rng.normal(size=(batch, horizon + 1, nu)) / np.sqrt((horizon + 1) * nu)
    initial = np.zeros((batch, horizon + 1, nx + nu))
    # Exact same explicit seed is supplied to both solvers.
    initial[:, :, :nx] = x0[:, None]
    return dict(
        x0=x0,
        reference=reference,
        diagonal=diagonal,
        costs=costs,
        probe=probe,
        initial=initial,
        dt=dt,
        nx=nx,
        nu=nu,
        horizon=horizon,
        control_limit=1.2,
        state_limit=20.0,
    )


def rhs_numpy(x, u, nonlinear):
    if nonlinear:
        return np.array([x[1], u[0] - 0.3 * np.sin(x[0]) - 0.1 * x[1]])
    return np.concatenate((x[3:6], x[6:9], u))


def physical_residuals(trajectory, case, nonlinear):
    nx, dt = case["nx"], case["dt"]
    x, u = trajectory[..., :nx], trajectory[..., nx:]
    predicted = np.empty_like(x[:, 1:])
    for b in range(len(x)):
        for k in range(case["horizon"]):
            state, control = x[b, k], u[b, k]
            k1 = rhs_numpy(state, control, nonlinear)
            k2 = rhs_numpy(state + dt / 2 * k1, control, nonlinear)
            k3 = rhs_numpy(state + dt / 2 * k2, control, nonlinear)
            k4 = rhs_numpy(state + dt * k3, control, nonlinear)
            predicted[b, k] = state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return {
        "initial_state": float(np.max(np.abs(x[:, 0] - case["x0"]))),
        "dynamics": float(np.max(np.abs(x[:, 1:] - predicted))),
        "state_bounds": float(np.maximum(np.abs(x) - case["state_limit"], 0).max()),
        "control_bounds": float(np.maximum(np.abs(u) - case["control_limit"], 0).max()),
    }
