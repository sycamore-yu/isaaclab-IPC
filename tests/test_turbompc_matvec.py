"""CPU oracles for the compile-only FP64 matrix/vector contraction."""

import math
import operator
from decimal import Decimal, localcontext

import pytest
import torch

from isaaclab_ipc.turbompc.solver import _matvec


def original(a, x):
    return (a @ x.unsqueeze(-1)).squeeze(-1)


@pytest.fixture
def compilation():
    # Exercise real Dynamo/fullgraph selection without CUDA or CPU code generation.
    torch._dynamo.reset()
    graphs = []

    def backend(graph, _inputs):
        graphs.append(graph)
        return graph.forward

    def compile_function(function):
        return torch.compile(function, backend=backend, fullgraph=True, dynamic=False)

    yield compile_function, graphs
    torch._dynamo.reset()


def decimal_dot(a, x):
    """Independent exact input conversion and 90-digit accumulation per dot."""
    matrix, vector = torch.broadcast_tensors(a, x.unsqueeze(-2))
    inner = matrix.shape[-1]
    rows = math.prod(matrix.shape[:-1])
    values, magnitudes = [], []
    with localcontext() as context:
        context.prec = 90
        for left, right in zip(matrix.reshape(rows, inner).tolist(), vector.reshape(rows, inner).tolist(), strict=True):
            products = [Decimal.from_float(lv) * Decimal.from_float(rv) for lv, rv in zip(left, right, strict=True)]
            values.append(float(sum(products, Decimal(0))))
            magnitudes.append(float(sum((abs(p) for p in products), Decimal(0))))
    shape = matrix.shape[:-1]
    expected = torch.tensor(values, dtype=torch.float64).reshape(shape)
    scale = torch.tensor(magnitudes, dtype=torch.float64).reshape(shape)
    bound = 16 * torch.finfo(torch.float64).eps * max(1, inner) * scale.clamp_min(1)
    return expected, bound


@pytest.mark.parametrize(
    ("a_shape", "x_shape", "layout"),
    [
        ((3, 5), (5,), "contiguous"),
        ((2, 3, 5), (5,), "contiguous"),
        ((2, 1, 3, 5), (4, 5), "contiguous"),
        ((2, 3, 5), (2, 5), "strided"),
        ((2, 3, 5), (2, 5), "expanded"),
        ((2, 3, 0), (2, 0), "contiguous"),
        ((2, 0, 5), (2, 5), "contiguous"),
        ((0, 3, 5), (0, 5), "contiguous"),
    ],
)
def test_compiled_contraction_matches_high_precision_and_owned_inputs(compilation, a_shape, x_shape, layout):
    compile_function, graphs = compilation
    generator = torch.Generator().manual_seed(5271)

    def values(shape):
        if layout == "strided":
            return torch.randn(*shape[:-1], 2 * shape[-1], generator=generator, dtype=torch.float64)[..., ::2]
        if layout == "expanded":
            return torch.randn(1, *shape[1:], generator=generator, dtype=torch.float64).expand(shape)
        return torch.randn(shape, generator=generator, dtype=torch.float64)

    a, x = values(a_shape), values(x_shape)
    before_a, before_x = a.clone(), x.clone()
    eager = _matvec(a, x)
    reference = original(a, x)
    torch.testing.assert_close(eager, reference, atol=0, rtol=0)
    actual = compile_function(_matvec)(a, x)
    expected, bound = decimal_dot(a, x)
    assert actual.shape == reference.shape == expected.shape
    assert actual.dtype == torch.float64
    assert torch.all((actual - expected).abs() <= bound)
    assert torch.all((reference - expected).abs() <= bound)
    assert any(n.target == operator.mul for graph in graphs for n in graph.graph.nodes)
    assert not any(n.target == operator.matmul for graph in graphs for n in graph.graph.nodes)
    torch.testing.assert_close(a, before_a, atol=0, rtol=0)
    torch.testing.assert_close(x, before_x, atol=0, rtol=0)


@pytest.mark.parametrize(
    ("a_shape", "x_shape"),
    [((2, 3), (1,)), ((2, 1), (3,)), ((2, 3, 4), (5, 4)), ((), (3,)), ((2, 3), ())],
)
def test_invalid_inner_batch_and_rank_do_not_become_broadcastable(compilation, a_shape, x_shape):
    compile_function, _ = compilation
    a, x = torch.ones(a_shape, dtype=torch.float64), torch.ones(x_shape, dtype=torch.float64)
    with pytest.raises(RuntimeError):
        original(a, x)
    with pytest.raises(RuntimeError):
        _matvec(a, x)
    with pytest.raises(RuntimeError):
        compile_function(_matvec)(a, x)


@pytest.mark.parametrize("dtype", [torch.float32, torch.complex128, torch.int32])
def test_other_dtypes_retain_original_operator_and_output_dtype(compilation, dtype):
    compile_function, graphs = compilation
    a = torch.tensor([[1, 2, 3], [-2, 4, 1]], dtype=dtype)
    x = torch.tensor([3, -2, 4], dtype=dtype)
    expected = original(a, x)
    actual = compile_function(_matvec)(a, x)
    assert actual.dtype == expected.dtype
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert any(n.target == operator.matmul for graph in graphs for n in graph.graph.nodes)


def test_mixed_dtypes_still_reject_and_special_ranks_keep_original_semantics(compilation):
    compile_function, graphs = compilation
    compiled = compile_function(_matvec)
    a, x = torch.ones((2, 3), dtype=torch.float64), torch.ones(3, dtype=torch.float32)
    with pytest.raises(RuntimeError):
        original(a, x)
    with pytest.raises(RuntimeError):
        compiled(a, x)
    for a, x in (
        (torch.arange(3, dtype=torch.float64), torch.ones(3, dtype=torch.float64)),
        (torch.ones((3, 1), dtype=torch.float64), torch.tensor(2.0, dtype=torch.float64)),
    ):
        torch.testing.assert_close(compiled(a, x), original(a, x), atol=0, rtol=0)
    assert any(n.target == operator.matmul for graph in graphs for n in graph.graph.nodes)


def test_sparse_layout_eager_path_is_unchanged():
    a = torch.tensor([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]], dtype=torch.float64).to_sparse()
    x = torch.tensor([2.0, 4.0, -1.0], dtype=torch.float64)
    torch.testing.assert_close(_matvec(a, x), original(a, x), atol=0, rtol=0)


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_results_are_not_sanitized(compilation, nonfinite):
    compile_function, _ = compilation
    a = torch.tensor([[1.0, 2.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float64)
    x = torch.tensor([1.0, nonfinite, 3.0], dtype=torch.float64)
    actual = compile_function(_matvec)(a, x)
    expected = original(a, x)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0, equal_nan=True)
    assert not torch.isfinite(actual).all()


def test_compiled_vmap_first_and_second_derivatives_match_bilinear_oracle(compilation):
    compile_function, graphs = compilation
    generator = torch.Generator().manual_seed(752)
    a = torch.randn((2, 3, 4), dtype=torch.float64, generator=generator).requires_grad_()
    x = torch.randn((2, 4), dtype=torch.float64, generator=generator).requires_grad_()
    da, dx = (
        torch.randn(a.shape, dtype=a.dtype, generator=generator),
        torch.randn(x.shape, dtype=x.dtype, generator=generator),
    )

    def objective(matrix, vector):
        value = torch.vmap(_matvec)(matrix, vector)
        return 0.5 * value.square().sum()

    loss = compile_function(objective)(a, x)
    ga, gx = torch.autograd.grad(loss, (a, x), create_graph=True)
    ha, hx = torch.autograd.grad((ga * da).sum() + (gx * dx).sum(), (a, x))
    value = torch.einsum("bmn,bn->bm", a, x)
    direction = torch.einsum("bmn,bn->bm", da, x) + torch.einsum("bmn,bn->bm", a, dx)
    expected_ga = torch.einsum("bm,bn->bmn", value, x)
    expected_gx = torch.einsum("bmn,bm->bn", a, value)
    expected_ha = torch.einsum("bm,bn->bmn", direction, x) + torch.einsum("bm,bn->bmn", value, dx)
    expected_hx = torch.einsum("bmn,bm->bn", da, value) + torch.einsum("bmn,bm->bn", a, direction)
    for actual, expected in zip((ga, gx, ha, hx), (expected_ga, expected_gx, expected_ha, expected_hx), strict=True):
        torch.testing.assert_close(actual, expected, atol=2e-13, rtol=2e-13)
    assert any(n.target == operator.mul for graph in graphs for n in graph.graph.nodes)
    assert not any(n.target == operator.matmul for graph in graphs for n in graph.graph.nodes)
