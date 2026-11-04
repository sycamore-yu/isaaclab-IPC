"""Lockstep ROS protocol helpers: no lost or doubled simulation steps."""

import pytest

from isaaclab_ipc.ros_sim_clock import expected_scan_step, packed_step


@pytest.mark.parametrize("step", [0, 1, 9, 10, 100, 10000])
@pytest.mark.parametrize("commanded", [False, True])
def test_step_request_round_trip(step, commanded):
    message = packed_step(step, commanded)
    assert message >> 1 == step
    assert bool(message & 1) is commanded


def test_lidar_scans_at_exact_simulation_ticks():
    assert [step for step in range(101) if expected_scan_step(step)] == list(range(0, 101, 10))
