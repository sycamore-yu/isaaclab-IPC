"""New AC-MPC contracts: node costs, immutable references and checkpoint identity."""

import copy
import math

import pytest
import torch
from tensordict import TensorDict

from isaaclab_ipc.rl.ctbr_actor import CTBRMPCActor
from isaaclab_ipc.rl.ctbr_observation import ctbr_feature_size, make_ctbr_observation
from isaaclab_ipc.turbompc.ctbr import CTBRConfig, CTBRQPBuilder


def observation(nodes=5, batch=2, faces=2, device="cpu"):
    cfg = CTBRConfig(cost_nodes=nodes)
    state = torch.zeros((batch, 10), dtype=torch.float64, device=device)
    state[:, 2], state[:, 3] = 1.0, 1.0
    reference = state.new_zeros((batch, nodes, 9))
    reference[..., 2] = 1.0
    reference[..., 0] = torch.linspace(0, 0.2, nodes, device=device)
    rates, goal = state.new_zeros((batch, 3)), state.new_tensor([1.0, 0.0, 1.0]).expand(batch, -1)
    planes = state.new_zeros((batch, nodes, faces, 4))
    mask = torch.zeros(planes.shape[:-1], device=device, dtype=torch.bool)
    if faces:
        planes[..., 0, 0], planes[..., 0, 3] = 1.0, -2.0
        mask[..., 0] = True
    result = make_ctbr_observation(
        state, rates, goal, reference, state.new_zeros((batch, 1)), planes, mask, ctbr_config=cfg,
    )
    return cfg, result


@pytest.mark.parametrize("nodes", [2, 5, 10])
def test_cost_node_and_transition_counts_are_explicit(nodes):
    cfg, obs = observation(nodes)
    builder = CTBRQPBuilder(cfg)
    data, guess = builder(obs, torch.zeros((2, 28 * nodes)))
    assert cfg.horizon == nodes - 1 and cfg.cost_dim == 28 * nodes
    assert data.D.shape == (2, nodes, 14, 14)
    assert data.E.shape == (2, nodes - 1, 14, 14)
    assert guess.shape == (2, nodes, 14)
    torch.testing.assert_close(guess[:, 0, :10], obs["ctbr_state"])
    assert obs["policy"].shape == (2, ctbr_feature_size(nodes))
    # Initial state is fixed by equality, not by duplicate velocity/SFC rows.
    assert torch.count_nonzero(data.G[:, 0, :3]) == 0
    assert torch.count_nonzero(data.G[:, 0, 7:]) == 0


def test_centered_cost_mapping_matches_independent_objective():
    cfg, obs = observation()
    builder = CTBRQPBuilder(cfg)
    rng = torch.Generator().manual_seed(712)
    logits = torch.randn((2, cfg.cost_dim), dtype=torch.float64, generator=rng)
    reference = torch.cat((obs["ctbr_reference"], obs["ctbr_control_reference"]), -1)
    d, q = builder.map_costs(reference, logits)
    z = reference + 0.1 * torch.randn(reference.shape, dtype=torch.float64, generator=rng)
    raw_d, raw_l = logits.reshape(2, 2, cfg.cost_nodes, 14).unbind(1)
    expected_d = builder.base_diagonal * (
        cfg.diagonal_floor + (1 - cfg.diagonal_floor) * (2 * torch.sigmoid(raw_d))
    )
    expected_l = builder.base_diagonal * builder.linear_scales * (2 * torch.sigmoid(raw_l) - 1)
    residual = z - reference
    centered = (0.5 * expected_d * residual.square() + expected_l * residual).sum((-1, -2))
    absolute = (0.5 * d * z.square() + q * z).sum((-1, -2))
    constant = (0.5 * d * reference.square() + q * reference).sum((-1, -2))
    torch.testing.assert_close(absolute - constant, centered, atol=1e-13, rtol=1e-13)
    torch.testing.assert_close(d, expected_d)


def test_each_node_and_coordinate_has_an_independent_diagonal_and_linear_head():
    cfg, obs = observation(nodes=2, batch=1)
    builder = CTBRQPBuilder(cfg)
    reference = torch.cat((obs["ctbr_reference"], obs["ctbr_control_reference"]), -1)
    logits = torch.zeros((1, cfg.cost_dim), dtype=torch.float64, requires_grad=True)
    diagonal, q = builder.map_costs(reference, logits)
    d_grad = torch.autograd.grad(diagonal[0, 1, 8], logits, retain_graph=True)[0]
    assert d_grad.nonzero().tolist() == [[0, 14 + 8]]
    # The residual linear coefficient is q + D*reference.
    l_grad = torch.autograd.grad((q + diagonal * reference)[0, 1, 8], logits)[0]
    assert l_grad.nonzero().tolist() == [[0, 28 + 14 + 8]]


def test_zero_logits_and_fixed_cost_are_identical():
    cfg, obs = observation()
    builder = CTBRQPBuilder(cfg)
    fixed, _ = builder(obs)
    learned, _ = builder(obs, torch.zeros((2, cfg.cost_dim), dtype=torch.float64))
    torch.testing.assert_close(learned.D, fixed.D, atol=0, rtol=0)
    torch.testing.assert_close(learned.q, fixed.q, atol=0, rtol=0)


def test_observation_owns_tensors_and_uses_acceleration_for_reference_attitude():
    cfg, base = observation()
    state, reference = base["ctbr_state"], base["reference"]
    reference[..., 6] = 2.0
    obs = make_ctbr_observation(
        state, base["body_rates"], base["goal"], reference, base["yaw_target"],
        base["planes"], base["plane_mask"], ctbr_config=cfg,
        warm_start_controls=base["ctbr_warm_start"],
    )
    frozen = {key: value.clone() for key, value in obs.items()}
    state.add_(1)
    reference.zero_()
    base["ctbr_warm_start"].fill_(100)
    for key in obs:
        torch.testing.assert_close(obs[key], frozen[key])
    assert obs["ctbr_reference"][..., 5].abs().min() > 0.05
    torch.testing.assert_close(obs["ctbr_reference"][..., 3:7].norm(dim=-1), torch.ones((2, 5), dtype=torch.float64))
    force = cfg.mass * (cfg.gravity**2 + 2**2) ** 0.5
    torch.testing.assert_close(obs["ctbr_control_reference"][..., 0], torch.full((2, 5), force, dtype=torch.float64))


def test_frozen_controls_warm_start_integrates_from_the_measured_state():
    cfg, obs = observation()
    builder = CTBRQPBuilder(cfg)
    obs["ctbr_state"][:, 7] = 0.7
    obs["ctbr_warm_start"][..., 0] *= 1.1
    obs["ctbr_warm_start"][..., 2] = 0.2
    _, guess = builder(obs)
    model = builder.dynamics()
    states = model.rollout(obs["ctbr_state"], obs["ctbr_warm_start"])
    torch.testing.assert_close(guess[..., :10], states)
    torch.testing.assert_close(guess[:, :-1, 10:], obs["ctbr_warm_start"])


@pytest.mark.parametrize("linesearch", ["merit", "none", None])
def test_new_actor_has_28n_outputs_and_rejects_wrong_checkpoint_semantics(linesearch):
    cfg, values = observation()
    obs = TensorDict(values, batch_size=[2])
    actor = CTBRMPCActor(
        obs, {"actor": ["policy"]}, "actor", 4, compile_kernels=False, sqp_linesearch=linesearch,
    )
    restored = CTBRMPCActor(
        obs, {"actor": ["policy"]}, "actor", 4, compile_kernels=False,
        sqp_linesearch="none" if linesearch == "merit" else "merit",
    )
    assert actor.mlp(actor.get_latent(obs)).shape == (2, cfg.cost_dim)
    assert not hasattr(actor, "ipc_problem")
    saved = copy.deepcopy(actor.state_dict())
    restored.load_state_dict(saved)
    assert saved["_extra_state"]["cost_outputs"] == 140
    assert restored.sqp_config.linesearch == ("none" if linesearch is None else linesearch)
    old_saved = copy.deepcopy(saved)
    old_saved["_extra_state"].pop("sqp_linesearch")
    restored.load_state_dict(old_saved)
    assert restored.solver_metadata()["sqp"]["linesearch"] == "merit"
    wrong = copy.deepcopy(saved)
    wrong["_extra_state"]["contract"] = "seven-head-ctbr"
    with pytest.raises(ValueError, match="checkpoint"):
        restored.load_state_dict(wrong)
    actor.close()
    restored.close()


def test_dense_sfc_preserves_every_face_and_exact_node_ownership():
    cfg, obs = observation(faces=69)
    # Include a real face beyond the old fixed 64-face budget.
    obs["planes"][0, 4, 68] = obs["planes"].new_tensor([0.0, 2.0, 0.0, -0.9])
    obs["plane_mask"][0, 4, 68] = True
    obs["planes"][~obs["plane_mask"]] = torch.nan
    builder = CTBRQPBuilder(cfg)
    data, _ = builder(obs)
    torch.testing.assert_close(data.G[0, 4, 7 + 68, :3], obs["planes"][0, 4, 68, :3])
    assert data.upper[0, 4, 7 + 68] == 0.9
    assert torch.isfinite(data.G).all()
    assert data.G[0, 3, 7 + 68].abs().sum() == 0
    assert torch.isposinf(data.upper[0, 3, 7 + 68])


def test_reference_quaternion_sign_is_physically_invariant():
    cfg, obs = observation()
    builder = CTBRQPBuilder(cfg)
    expected, _ = builder(obs)
    obs["ctbr_reference"][:, ::2, 3:7] *= -1
    actual, _ = builder(obs)
    torch.testing.assert_close(actual.q, expected.q, atol=0, rtol=0)


def test_articulation_mass_includes_rotor_links():
    from types import SimpleNamespace

    from isaaclab_ipc.tasks.ctbr_env import checked_articulation_mass

    link_masses = torch.tensor([[0.746, 0.001, 0.001, 0.001, 0.001]])
    robot = SimpleNamespace(
        num_instances=1, num_bodies=5,
        data=SimpleNamespace(body_mass=SimpleNamespace(torch=link_masses)),
    )
    torch.testing.assert_close(checked_articulation_mass(robot, 0.75), torch.tensor([0.75]))
    link_masses[0, 0] += 0.01
    with pytest.raises(ValueError, match="masses differ"):
        checked_articulation_mass(robot, 0.75)

def test_frontend_and_ctbr_share_physical_parameters():
    from isaaclab_ipc.pathcover_native import PathCoverConfig
    from isaaclab_ipc.tasks.ctbr_env import pathcover_config_for_ctbr

    model = CTBRConfig(mass=0.9, gravity=9.7)
    frontend = pathcover_config_for_ctbr(model, {"max_velocity": 4.0, "body_rate_max": 2.0})
    assert frontend.mass == model.mass and frontend.gravity == model.gravity
    assert frontend.thrust_max == pytest.approx(model.mass * model.specific_thrust_limits[1])
    assert frontend.thrust_min == model.mass * model.specific_thrust_limits[0]
    assert frontend.nominal_thrust_min == pytest.approx(2.0 / 0.716 * model.mass)
    assert frontend.body_yaw_rate_max == model.body_rate_limits[2]
    assert frontend.max_velocity == 4.0 and frontend.body_rate_max == 2.0
    assert frontend.nominal_velocity == 3.0
    default = pathcover_config_for_ctbr(model, {})
    assert default.max_velocity == 10.0 and default.nominal_velocity == 3.0
    assert default.tilt_max == math.pi / 2
    assert default.max_corridors == 4 and default.safety_margin == 0.0
    assert default.max_iterations == 2000
    assert default.pieces_per_polytope == 1
    assert default.position_penalty_aggregation == "continuous_max"
    assert default.geometry_model == "box_ellipsoid"
    standalone = PathCoverConfig()
    assert standalone.max_corridors == 6 and standalone.safety_margin == 0.15
    assert standalone.max_iterations == 1000
    assert standalone.position_penalty_aggregation == "integral"
    assert standalone.geometry_model == "inflated_voxels"
    assert default.penalty_weights == (1e5, 100.0, 1e4, 1e4, 1e4)
    assert default.body_rate_max == 3.0 and default.reference_body_rate_margin == 0.01
    assert model.body_rate_limits == (3.0, 3.0, 1.5)
    narrow = CTBRConfig(mass=0.9, specific_thrust_limits=(4.0, 12.0))
    restricted = pathcover_config_for_ctbr(narrow, {})
    assert restricted.thrust_min == pytest.approx(3.6)
    assert restricted.nominal_thrust_min == restricted.thrust_min
    assert restricted.thrust_max == pytest.approx(10.8)
    stress = pathcover_config_for_ctbr(model, {
        "nominal_velocity": 6.0, "thrust_min": 0.0, "nominal_thrust_min": 0.0,
        "thrust_max": model.mass * model.specific_thrust_limits[1],
        "tilt_max": 0.8, "penalty_weights": (1e4, 1e3, 1e4, 1e4, 1e4),
        "max_corridors": 6, "safety_margin": 0.15, "position_penalty_aggregation": "integral",
        "geometry_model": "inflated_voxels", "max_iterations": 500,
    })
    assert stress.nominal_velocity == 6.0 and stress.thrust_min == 0.0
    assert stress.nominal_thrust_min == 0.0
    assert stress.thrust_max == model.mass * model.specific_thrust_limits[1]
    assert stress.tilt_max == 0.8
    assert stress.max_corridors == 6 and stress.safety_margin == 0.15
    assert stress.position_penalty_aggregation == "integral"
    assert stress.geometry_model == "inflated_voxels"
    assert stress.max_iterations == 500
    conservative = pathcover_config_for_ctbr(model, {
        "thrust_min": 2.0 / 0.716 * model.mass, "thrust_max": 10.0 / 0.716 * model.mass,
        "tilt_max": 0.7854,
    })
    assert conservative.thrust_min == pytest.approx(2.0 / 0.716 * model.mass)
    assert conservative.nominal_thrust_min == conservative.thrust_min
    assert conservative.thrust_max == pytest.approx(10.0 / 0.716 * model.mass)
    assert conservative.tilt_max == 0.7854
    with pytest.raises(ValueError, match="thrust_min"):
        pathcover_config_for_ctbr(narrow, {"thrust_min": 3.5})
    with pytest.raises(ValueError, match="nominal_thrust_min"):
        pathcover_config_for_ctbr(narrow, {"nominal_thrust_min": 3.5})
    with pytest.raises(ValueError, match="mass"):
        pathcover_config_for_ctbr(model, {"mass": 0.75})
    with pytest.raises(ValueError, match="body_yaw_rate_max"):
        pathcover_config_for_ctbr(model, {"body_yaw_rate_max": 2.0})


@pytest.mark.parametrize("benchmark, physical_speed", [("high_speed", 10.0), ("dense", 8.0)])
def test_marsim_keeps_cruise_target_distinct_from_physical_speed(benchmark, physical_speed):
    from isaaclab_ipc.benchmark_config import load_case
    from isaaclab_ipc.ros_ctbr_policy import benchmark_frontend_config

    model = CTBRConfig()
    frontend = benchmark_frontend_config(load_case(benchmark), model)
    assert frontend.max_velocity == physical_speed
    assert frontend.nominal_velocity == 3.0
    assert frontend.thrust_min == model.mass * model.specific_thrust_limits[0]
    assert frontend.nominal_thrust_min == pytest.approx(2.0 / 0.716 * model.mass)
    assert frontend.thrust_max == pytest.approx(model.mass * model.specific_thrust_limits[1])
    assert frontend.tilt_max == math.pi / 2
    assert frontend.max_corridors == 4 and frontend.safety_margin == 0.0
    assert frontend.max_iterations == 2000
    assert frontend.pieces_per_polytope == 1
    assert frontend.position_penalty_aggregation == "continuous_max"
    assert frontend.geometry_model == "box_ellipsoid"
    assert frontend.penalty_weights == (1e5, 100.0, 1e4, 1e4, 1e4)
    assert frontend.body_rate_max == 3.0 and frontend.reference_body_rate_margin == 0.01
    assert model.body_rate_limits == (3.0, 3.0, 1.5)
    stress = benchmark_frontend_config(load_case(benchmark), model, reference_speed=6.0)
    assert stress.max_velocity == physical_speed and stress.nominal_velocity == 6.0
    narrow = CTBRConfig(specific_thrust_limits=(4.0, 12.0))
    restricted = benchmark_frontend_config(load_case(benchmark), narrow)
    assert restricted.thrust_min == pytest.approx(4.0 * narrow.mass)
    assert restricted.nominal_thrust_min == restricted.thrust_min
    assert restricted.thrust_max == pytest.approx(12.0 * narrow.mass)
    explicit = benchmark_frontend_config(load_case(benchmark), model, nominal_thrust_min=0.0)
    assert explicit.nominal_thrust_min == explicit.thrust_min == 0.0
    with pytest.raises(ValueError, match="nominal_thrust_min"):
        benchmark_frontend_config(load_case(benchmark), narrow, nominal_thrust_min=2.0)


def test_single_environment_snapshots_have_canonical_strides():
    cfg, first = observation(batch=1, faces=0)
    physical = torch.zeros((1, 19), dtype=torch.float64)
    physical[:, :10] = first["ctbr_state"]
    controls = first["ctbr_control_reference"][:, :-1]
    assert physical[:, :10].stride() == (19, 1)
    assert controls.stride() == (20, 4, 1)
    replay = make_ctbr_observation(
        physical[:, :10], first["body_rates"], first["goal"], first["reference"],
        first["yaw_target"], first["planes"], first["plane_mask"],
        ctbr_config=cfg, warm_start_controls=controls,
    )
    for name, value in replay.items():
        assert value.stride() == torch.empty(value.shape).stride(), name
    assert replay["ctbr_state"].stride() == (10, 1)
    assert replay["ctbr_warm_start"].stride() == (16, 4, 1)


def test_diagonal_floor_cannot_reverse_the_positive_cost_mapping():
    with pytest.raises(ValueError, match="diagonal_floor"):
        CTBRConfig(diagonal_floor=3.0)
