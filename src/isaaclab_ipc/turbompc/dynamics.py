"""Discrete prediction models; independent of the simulation backend."""

from collections.abc import Callable

import torch


class LinearDynamics:
    """One-step map x_next = matrix @ [x, u] + offset."""

    def __init__(self, matrix: torch.Tensor, *, nx: int, offset: torch.Tensor | None = None):
        if matrix.ndim != 2 or matrix.shape[0] != nx or matrix.shape[1] <= nx:
            raise ValueError("Expected a dynamics matrix [nx, nx + nu]")
        self.matrix = matrix
        self.offset = offset
        self.nx = nx

    def __call__(self, z):
        value = z @ self.matrix.T
        return value if self.offset is None else value + self.offset

    def linearize(self, z):
        return self(z), self.matrix.expand(*z.shape[:-1], *self.matrix.shape)

    def lagrangian_hessian(self, z, multipliers):
        return torch.zeros((*z.shape, z.shape[-1]), dtype=z.dtype, device=z.device)


class AutoDiffDynamics:
    """A smooth one-step function with batched Jacobians and exact second derivatives.

    ``step`` receives a single concatenated [state, control] vector. Physical
    constants are captured by the function; this layer does not learn them.
    """

    def __init__(self, step: Callable[[torch.Tensor], torch.Tensor], *, nx: int):
        self.nx = nx
        self._step = torch.vmap(step)

        def value_with_auxiliary(z):
            value = step(z)
            return value, value

        # jacrev already evaluates the step. Return that value as auxiliary data
        # instead of launching the nonlinear dynamics a second time per node.
        self._value_and_jacobian = torch.vmap(torch.func.jacrev(value_with_auxiliary, has_aux=True))
        # Reverse-over-reverse avoids forward-AD inference-tensor failures in
        # Torch 2.11 compiled backward, while retaining the exact Hessian.
        gradient = torch.func.jacrev(lambda z, lam: (step(z) * lam).sum(), argnums=0)
        self._hessian = torch.vmap(torch.func.jacrev(gradient, argnums=0))

    def __call__(self, z):
        return self._step(z.reshape(-1, z.shape[-1])).reshape(*z.shape[:-1], self.nx)

    def linearize(self, z):
        flat = z.reshape(-1, z.shape[-1])
        jacobian, value = self._value_and_jacobian(flat)
        return value.reshape(*z.shape[:-1], self.nx), jacobian.reshape(*z.shape[:-1], self.nx, z.shape[-1])

    def lagrangian_hessian(self, z, multipliers):
        return self._hessian(z.reshape(-1, z.shape[-1]), multipliers.reshape(-1, self.nx)).reshape(
            *z.shape, z.shape[-1]
        )


def rk4_step(rhs, z, nx: int, dt: float):
    """Fourth-order Runge-Kutta with control held constant over one interval."""
    x, u = z[:nx], z[nx:]
    k1 = rhs(x, u)
    k2 = rhs(x + 0.5 * dt * k1, u)
    k3 = rhs(x + 0.5 * dt * k2, u)
    k4 = rhs(x + dt * k3, u)
    return x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
