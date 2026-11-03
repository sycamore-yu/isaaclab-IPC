"""CPU-only affine QP metadata checks against the physical one-step map."""

import pytest
import torch

from isaaclab_ipc.turbompc import AutoDiffDynamics, LinearDynamics, MPCInputs, TorchMPC


def problem():
    batch, nodes, nx, width = 2, 4, 2, 3
    matrix = torch.tensor([[1.0, 0.2, 0.1], [-0.1, 0.9, 0.25]], dtype=torch.float64, device="cpu")
    identity = torch.eye(width, dtype=torch.float64, device="cpu").expand(batch, nodes, -1, -1)
    inputs = MPCInputs(
        initial_state=torch.tensor([[0.3, -0.2], [-0.1, 0.4]], dtype=torch.float64),
        D=identity,
        E=torch.zeros((batch, nodes - 1, width, width), dtype=torch.float64),
        q=torch.zeros((batch, nodes, width), dtype=torch.float64),
        G=identity,
        lower=torch.full((batch, nodes, width), -5.0, dtype=torch.float64),
        upper=torch.full((batch, nodes, width), 5.0, dtype=torch.float64),
    )
    query = torch.linspace(-0.8, 0.9, batch * nodes * width, dtype=torch.float64).reshape(batch, nodes, width)
    return matrix, nx, inputs, query


def assert_physical_equalities(model, inputs, query):
    """The QP residual must equal f(z[k])-x[k+1], at any expansion point."""
    solver = TorchMPC(model)
    try:
        expected = model(query[:, :-1]) - query[:, 1:, : model.nx]
        initial = query[:, 0, : model.nx] - inputs.initial_state
        for expansion in (torch.zeros_like(query), query.flip(1) * 3):
            qp = solver._qp(inputs, expansion)
            # The native QP contract is float64, including offsets which the
            # physical map promotes from integer or float32 Tensor inputs.
            assert all(getattr(qp, name).dtype == torch.float64 for name in qp.__dataclass_fields__)
            assert all(getattr(qp, name).device.type == "cpu" for name in qp.__dataclass_fields__)
            assert qp.c_dyn.shape == expected.shape
            residual = (
                torch.einsum("btij,btj->bti", qp.A_minus, query[:, :-1])
                + torch.einsum("btij,btj->bti", qp.A_plus, query[:, 1:])
                - qp.c_dyn
            )
            initial_residual = torch.einsum("bij,bj->bi", qp.A0, query[:, 0]) - qp.c0
            torch.testing.assert_close(residual, expected, atol=1e-12, rtol=0)
            torch.testing.assert_close(initial_residual, initial, atol=1e-12, rtol=0)
    finally:
        solver.close()


@pytest.mark.parametrize("offset_kind", ["none", "float64_vector", "float32_vector", "integer_scalar", "batched"])
def test_affine_qp_matches_physical_equalities_for_supported_offsets(offset_kind):
    matrix, nx, inputs, query = problem()
    offsets = {
        "none": None,
        "float64_vector": torch.tensor([0.125, -0.25], dtype=torch.float64),
        "float32_vector": torch.tensor([0.125, -0.25], dtype=torch.float32),
        "integer_scalar": torch.tensor(1, dtype=torch.int64),
        "batched": torch.tensor([[[0.125, -0.25]], [[-0.5, 0.375]]], dtype=torch.float64),
    }
    assert_physical_equalities(LinearDynamics(matrix, nx=nx, offset=offsets[offset_kind]), inputs, query)


def test_linear_dynamics_subclass_uses_its_overridden_linearization():
    class ShiftedModel(LinearDynamics):
        calls = 0

        def __call__(self, z):
            return super().__call__(z) + 0.3 * z[..., : self.nx]

        def linearize(self, z):
            self.calls += 1
            return AutoDiffDynamics(self, nx=self.nx).linearize(z)

    matrix, nx, inputs, query = problem()
    model = ShiftedModel(matrix, nx=nx, offset=torch.tensor([0.125, -0.25], dtype=torch.float64))
    assert_physical_equalities(model, inputs, query)
    assert model.calls == 2
