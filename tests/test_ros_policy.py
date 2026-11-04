"""Numerical deployment contracts; live MARSIM runs remain the acceptance test."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from isaaclab_ipc.corridor_storage import expand_corridors
from isaaclab_ipc.frontend import PlanningContext
from isaaclab_ipc.mpc import MPC, Config, dynamics
from isaaclab_ipc.planner import Planner
from isaaclab_ipc.rl.actor import flatness_torch
from isaaclab_ipc.ros_control import (
    IdealCommand,
    accept_plan,
    check_plan_constraints,
    ctbr_from_jerk,
    ctbr_from_reference,
    select_jerk,
    sequence_reference,
)
from isaaclab_ipc.ros_policy import observation


def test_ros_features_equal_the_actual_training_expression():
    """Evaluate the source training expression without loading a simulator app."""
    import isaaclab_ipc

    physical = torch.arange(18, dtype=torch.float32).reshape(1, 18) / 7
    goal = torch.tensor([[1.1, -2.3, 4.7]], dtype=torch.float32)
    reference = torch.arange(135, dtype=torch.float64).reshape(1, 15, 9) / 13
    face = np.array([[1., 2., 0., -3.], [-1., 0., 0., -1.]])
    corridors = [face if i % 3 else np.empty((0, 4)) for i in range(15)]
    margins = [[min(-(f[:3] @ reference[0, i, :3].numpy() + f[3]) / np.linalg.norm(f[:3])
                    for f in faces) if len(faces) else 10. for i, faces in enumerate(corridors)]]
    braking = [[True]]

    def tensor(value, dtype=torch.float64):
        return torch.as_tensor(np.asarray(value), dtype=dtype)

    source = Path(isaaclab_ipc.__file__).parent / "tasks/mpc_env.py"
    tree = ast.parse(source.read_text())
    expression = next(n.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == "features" for target in n.targets))
    expected = eval(compile(ast.Expression(expression), str(source), "eval"), dict(
        torch=torch, physical=physical, reference=reference, self=SimpleNamespace(goal=goal),
        tensor=tensor, margins=margins, braking=braking,
    ))
    context = PlanningContext(physical[0, :9].double().numpy(), reference[0].numpy(), corridors, True)
    actual = observation(physical.numpy(), goal[0].numpy(), context, braking=True)
    assert actual["policy"].shape == (1, 124)
    torch.testing.assert_close(actual["policy"], expected, rtol=0, atol=0)
    torch.testing.assert_close(actual["mpc_state"], physical[:, :9].double(), rtol=0, atol=0)
    torch.testing.assert_close(actual["reference"], reference, rtol=0, atol=0)
    assert bool(actual["braking"][0, 0])
    assert bool(actual["sfc_valid"][0, 0])
    dense, mask = expand_corridors(actual)
    for i, faces in enumerate(corridors):
        np.testing.assert_array_equal(dense[0, i, mask[0, i]].numpy(), faces[::-1])


def test_empty_frontend_context_preserves_recovery_inputs():
    physical = np.r_[[1., 2., 3., .4, .5, .6, .7, .8, .9], np.eye(3).ravel()]
    context = PlanningContext(physical[:9], np.zeros((15, 9)), [], False)
    target = np.array([1., 2., 4.])
    obs = observation(physical, [5., 6., 7.], context, yaw_target=.3, climb_target=target)
    assert not bool(obs["sfc_valid"][0, 0])
    assert bool(obs["climb_active"][0, 0])
    np.testing.assert_array_equal(obs["climb_target"][0].numpy(), target)
    assert float(obs["yaw_target"][0, 0]) == .3
    assert obs["plane_mask"].sum() == 0
    assert obs["corridor_index"].shape == (1, 15)
    assert torch.all(obs["corridor_index"] == -1)


def test_sequence_execution_clock_preserves_the_source_horizon():
    mpc = MPC(Config.load("avoid"))
    origin = 1_793_768_000_000_000_000
    sequence = np.arange(45, dtype=float).reshape(15, 3) - 20
    plan = dict(source_stamp_ns=origin, accepted_stamp_ns=origin, sequence=sequence,
                braking=False, sfc_valid=True, climb_target=None)
    state = np.zeros(9)
    state[3] = 1.0
    for elapsed, index in ((0, 0), (99_999_999, 0), (100_000_000, 1), (340_000_000, 3), (1_499_999_999, 14)):
        jerk, actual_index, braking, expired = select_jerk(plan, state, origin + elapsed, mpc)
        np.testing.assert_array_equal(jerk, sequence[index])
        assert actual_index == index
        assert not braking and not expired
    jerk, index, braking, expired = select_jerk(plan, state, origin + 1_500_000_000, mpc)
    assert index == 15 and braking and expired
    np.testing.assert_array_equal(jerk, [-20.0, 0.0, 0.0])


def test_delayed_acceptance_does_not_skip_unexecuted_acceleration():
    # Real checkpoint solution from padding_equivalence.npz. It accelerates
    # first, then uses negative jerk; source-time indexing skips that acceleration.
    sequence = np.array([
        [35.69299617046498, .18240948267993412, .09429317342325093],
        [7.227119616996953, .005541445163717798, .0028645465255485356],
        [-.6632227936397032, -.04295216347517047, -.02220331827032606],
        [-6.062132739430585, -.06374023131094245, -.03294932147571943],
        [-9.36816961743348, -.0653950615977789, -.0338047550690294],
        [-11.039754484677255, -.055252608537089284, -.02856180348922125],
        [-11.507488513394847, -.03921126184384441, -.020269529077464148],
        [-11.108523244726731, -.021684155691209493, -.011209219077254216],
        [-10.047514923974534, -.005684562224153374, -.002938528215541736],
        [-8.385610195386164, .006992042504234419, .0036144057137171073],
        [-6.056635699795584, .015530525529160039, .008028214956592306],
        [-2.907891682382328, .01984173505287896, .010256814157201298],
        [1.2386565758338464, .020322217641275866, .010505190652594693],
        [6.510733685880029, .01764926528501992, .009123457880934006],
        [25.80733743488903, .02524476846537281, .013049811314426106],
    ])
    planner = Planner(Config.load("avoid"), np.array([2., 0., 1.]))
    state = np.zeros(9)
    source = 1_793_768_000_000_000_000
    accepted = source + 240_000_000
    plan = dict(source_stamp_ns=source, accepted_stamp_ns=accepted, sequence=sequence,
                braking=False, sfc_valid=True, climb_target=None)
    _, original_jerk, _ = planner.consume_state(state, sequence.ravel(), "solved", accepted * 1e-9)
    jerk, index, braking, expired = select_jerk(plan, state, accepted, planner.mpc)
    np.testing.assert_array_equal(jerk, original_jerk)
    assert index == 0 and jerk[0] > 0 and not braking and not expired
    assert sequence[(accepted - source) // 100_000_000, 0] < 0
    jerk, index, _, _ = select_jerk(plan, state, accepted + 100_000_000, planner.mpc)
    assert index == 1
    np.testing.assert_array_equal(jerk, sequence[1])
    # Acceptance cannot renew an old source state beyond its original horizon.
    _, _, braking, expired = select_jerk(plan, state, source + 1_500_000_000, planner.mpc)
    assert braking and expired


def test_recovery_keeps_the_original_prediction_step_with_fresh_feedback():
    mpc = MPC(Config.load("avoid"))
    state = np.array([1., 2., 3., .4, .5, .6, .7, .8, .9])
    jerk = np.array([2., -3., 1.])
    rotation = np.eye(3)
    limits = (0., 30.)
    actual = ctbr_from_jerk(state, rotation, jerk, .3, mpc, limits, 10.)
    predicted = mpc.a @ state + mpc.b @ jerk
    expected = flatness_torch(torch.tensor(predicted[None, 6:]), torch.tensor(jerk[None]),
                              torch.tensor(rotation[None]), torch.tensor([.3], dtype=torch.float64), limits, 10.)
    np.testing.assert_allclose(actual, expected[0].numpy(), rtol=0, atol=1e-14)
    # Training stores R in float32, while ROS Rotation.as_matrix returns float64.
    from_training_rotation = ctbr_from_jerk(state, rotation.astype(np.float32), jerk, .3, mpc, limits, 10.)
    np.testing.assert_array_equal(from_training_rotation, actual)
    a_control, b_control = dynamics(.01)
    incorrect = flatness_torch(torch.tensor((a_control @ state + b_control @ jerk)[None, 6:]),
                               torch.tensor(jerk[None]), torch.tensor(rotation[None]),
                               torch.tensor([.3], dtype=torch.float64), limits, 10.)
    assert not np.allclose(actual, incorrect[0].numpy())
    state[8] += 1.
    changed = ctbr_from_jerk(state, rotation, jerk, .3, mpc, limits, 10.)
    assert changed[0] != actual[0]


def test_reference_knots_match_original_mpc_and_interpolate_with_jerk():
    mpc = MPC(Config.load("avoid"))
    state = np.arange(9, dtype=float) / 10
    sequence = np.arange(45, dtype=float).reshape(15, 3) / 30 - .5
    origin = 1_793_768_000_000_000_000
    plan = accept_plan(dict(sequence=sequence), state, origin, mpc)
    expected = (mpc.M @ state + mpc.C @ sequence.ravel()).reshape(15, 9)
    np.testing.assert_allclose(plan["states"][1:], expected, rtol=0, atol=1e-13)
    for index in range(1, 16):
        sampled = sequence_reference(plan, origin + index * 100_000_000, mpc)
        np.testing.assert_array_equal(sampled, plan["states"][index])
    a, b = dynamics(.04)
    inside = sequence_reference(plan, origin + 340_000_000, mpc)
    np.testing.assert_array_equal(inside, a @ plan["states"][3] + b @ sequence[3])


def test_cached_jerk_does_not_repeatedly_advance_measured_acceleration():
    """An ideal vertical force response isolates the multi-rate amplification."""
    mpc = MPC(Config.load("avoid"))
    initial = np.zeros(9)
    sequence = np.tile([0., 0., 1.], (15, 1))
    plan = accept_plan(dict(sequence=sequence), initial, 0, mpc)
    old_state = initial.copy()
    for tick in range(1, 11):
        old_command = ctbr_from_jerk(old_state, np.eye(3), sequence[0], 0., mpc, (0., 30.), None)
        old_state[8] = old_command[0] - 9.81
        reference = sequence_reference(plan, tick * 10_000_000, mpc)
        command = ctbr_from_reference(reference[6:], np.eye(3), sequence[0], 0., (0., 30.), None)
    assert abs(old_state[8] - 1.0) < 1e-12
    assert abs(command[0] - 9.81 - .1) < 1e-12
    expected_at_model_step = mpc.a @ initial + mpc.b @ sequence[0]
    np.testing.assert_array_equal(reference, expected_at_model_step)


def test_ideal_command_advances_wall_time_despite_repeated_old_odometry():
    command = IdealCommand([0., 0., 0.])
    measured = np.array([1., 2., 3., .4, .5, .6, .7, .8, .9])
    jerk = np.array([2., -3., 1.])
    origin = 1_793_768_000_000_000_000
    command.advance(measured, jerk, origin)
    for elapsed in (10_000_000, 35_000_000, 70_000_000, 110_000_000):
        actual = command.advance(measured, jerk, origin + elapsed)
    a, b = dynamics(.11)
    np.testing.assert_allclose(actual, a @ measured + b @ jerk, rtol=0, atol=1e-14)


def test_ideal_command_integrates_the_jerk_that_was_actually_published():
    command = IdealCommand([0., 0., 0.])
    measured = np.zeros(9)
    first, second = np.array([2., 0., 0.]), np.array([-4., 0., 0.])
    command.advance(measured, first, 0)
    at_switch = command.advance(measured, second, 50_000_000)
    a, b = dynamics(.05)
    np.testing.assert_allclose(at_switch, b @ first, rtol=0, atol=1e-14)
    after_switch = command.advance(measured, second, 100_000_000)
    np.testing.assert_allclose(after_switch, a @ (b @ first) + b @ second, rtol=0, atol=1e-14)


def test_ideal_odometry_acceleration_matches_its_executed_command():
    command = IdealCommand([0., 0., 0.], history_size=8)
    measured = np.r_[[0., -50., 1.5], np.zeros(6)]
    jerk = np.array([50., -50., 5.])
    first = command.advance(measured, jerk, 0)
    delayed = command.advance(measured, jerk, 10_000_000)
    latest = command.advance(measured, jerk, 40_000_000)
    np.testing.assert_array_equal(command.observed_acceleration(first[:6]), first[6:])
    np.testing.assert_array_equal(command.observed_acceleration(delayed[:6]), delayed[6:])
    assert np.linalg.norm(latest[6:] - delayed[6:]) > 2.
    # A newer published command must not alter repeated delayed odometry.
    latest = command.advance(measured, jerk, 50_000_000)
    np.testing.assert_array_equal(command.observed_acceleration(delayed[:6]), delayed[6:])
    np.testing.assert_array_equal(command.observed_acceleration(latest[:6]), latest[6:])


def test_acceptance_constraints_preserve_the_original_problem_and_actual_pva():
    mpc = MPC(Config.load("avoid"))
    faces = np.array([[1., 0., 0., -5.], [-1., 0., 0., -5.]])
    plan = dict(sequence=np.zeros((15, 3)), corridors=[faces], corridor_index=np.zeros(15, dtype=int),
                constraint_eps_abs=1e-5, constraint_eps_rel=0.)
    state = np.zeros(9)
    state[3] = .4
    check = check_plan_constraints(plan, state, mpc)
    _, _, matrix, lower, upper, _ = mpc.problem(state, np.zeros((15, 9)), [faces] * 15)
    value = matrix @ plan["sequence"].ravel()
    original = np.maximum(np.maximum(lower - value, value - upper), 0).max()
    assert check["feasible"] and check["violation"] == original
    # A clipped-state check incorrectly accepts this trajectory. Its first
    # actual acceleration remains 21, beyond the unchanged component limit 20.
    state[:] = 0
    state[6] = 21.
    plan["corridor_index"][:] = -1
    plan["sequence"][1:5, 0] = -50.
    _, _, matrix, lower, upper, _ = mpc.problem(state, np.zeros((15, 9)), [np.empty((0, 4))] * 15)
    value = matrix @ plan["sequence"].ravel()
    assert np.maximum(np.maximum(lower - value, value - upper), 0).max() == 0
    check = check_plan_constraints(plan, state, mpc)
    assert not check["feasible"]
    assert check["violation"] == 1. and check["constraint"] == "acceleration"


def test_acceptance_rejects_the_recorded_v4_first_stage_at_its_actual_start():
    mpc = MPC(Config.load("avoid"))
    # ideal-acmpc-async-v4 plan 345. Only the first stage is needed to prove
    # infeasibility: no later unknown jerk can repair the first predicted knot.
    state = np.array([-3.0172692770875664, -9.299468176369349, 2.3830702825773726,
                      .6429373766727359, -3.4290622057496654, -1.6061689539639425,
                      21.4203259985403, -18.83027546561187, -11.400627676289272])
    sequence = np.zeros((15, 3))
    sequence[0] = [50., 50., -8.905287731201765]
    plan = dict(sequence=sequence, corridors=[], corridor_index=np.full(15, -1),
                constraint_eps_abs=1e-5, constraint_eps_rel=0.)
    first = mpc.a @ state + mpc.b @ sequence[0]
    assert first[6] > 26.4 and first[8] < -12.29
    assert not check_plan_constraints(plan, state, mpc)["feasible"]


def test_acceptance_reuses_sfc_planes_and_the_actor_primal_tolerance():
    mpc = MPC(Config.load("avoid"))
    faces = np.array([[1., 0., 0., -1.]])
    plan = dict(sequence=np.zeros((15, 3)), corridors=[faces], corridor_index=np.zeros(15, dtype=int),
                constraint_eps_abs=1e-5, constraint_eps_rel=0.)
    state = np.zeros(9)
    state[0] = 1.
    assert check_plan_constraints(plan, state, mpc)["feasible"]
    state[0] += .01
    check = check_plan_constraints(plan, state, mpc)
    assert not check["feasible"] and check["constraint"] == "sfc"
    np.testing.assert_allclose(check["violation"], .01, rtol=0, atol=1e-15)
    state[:] = 0
    plan["corridor_index"][:] = -1
    plan["sequence"][:2, 0] = [50. + 5e-6, -50.]
    assert check_plan_constraints(plan, state, mpc)["feasible"]
    plan["sequence"][0, 0] = 50. + 2e-5
    check = check_plan_constraints(plan, state, mpc)
    assert not check["feasible"] and check["constraint"] == "jerk"
