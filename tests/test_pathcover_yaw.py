"""Native yaw integration, its derivative, and exact failure-candidate capture."""

from types import SimpleNamespace

import numpy as np
import pytest

from isaaclab_ipc.pathcover_native import (
    NativePathCoverFrontend,
    PathCoverConfig,
    PathCoverPlanningError,
    PathCoverState,
    PathCoverTrajectory,
)


def native_module():
    return pytest.importorskip("isaaclab_ipc._pathcover_native")


def test_hermite_heading_derivative_and_endpoint_continuity():
    native_module()
    coefficients = np.zeros((1, 3, 6))
    coefficients[0, 0, 3:5] = (4.0, -3.0)
    coefficients[0, 1, 4] = 0.2
    trajectory = PathCoverTrajectory(coefficients, np.array([0.3]), initial_yaw=np.pi / 2,
                                    body_yaw_rate_max=1.5)
    times = np.linspace(0.0001, trajectory.duration - 0.0001, 1001)
    epsilon = 1e-7
    derivative = (trajectory.sample(times + epsilon).yaw - trajectory.sample(times - epsilon).yaw) / (2 * epsilon)
    np.testing.assert_allclose(trajectory.sample(times).yaw_rate, derivative, atol=1e-7, rtol=1e-7)
    knots = trajectory._yaw_times[1:-1]
    np.testing.assert_allclose(trajectory.sample(knots).yaw_rate, trajectory._yaw_rates[1:-1], atol=1e-12)
    np.testing.assert_allclose(trajectory.sample(knots - 1e-10).yaw_rate,
                               trajectory.sample(knots + 1e-10).yaw_rate, atol=1e-5)
    assert trajectory.sample(-0.1).yaw_rate == 0
    assert trajectory.sample(trajectory.duration + 0.1).yaw_rate == 0

    # Historical saved plans retain their recorded piecewise-linear profile.
    object.__delattr__(trajectory, "_yaw_rates")
    np.testing.assert_array_equal(trajectory.sample(times).yaw,
                                  np.interp(times, trajectory._yaw_times, trajectory._yaw_values))


def test_native_yaw_distinguishes_domain_failure_from_invalid_input():
    native = native_module()
    coefficients = np.zeros((1, 3, 6))
    args = (np.ones(1), 0.0, 1.5, 0.02, 0.1, 9.81)
    coefficients[0, 2, 3] = -9.81  # Downward total force throughout the polynomial.
    with pytest.raises(PathCoverPlanningError, match="upright"):
        native.bounded_body_yaw_profile(coefficients, *args)
    coefficients[0, 2, 3] = np.nan
    with pytest.raises(ValueError, match="finite"):
        native.bounded_body_yaw_profile(coefficients, *args)
    with pytest.raises(ValueError, match="heading input"):
        native.bounded_body_yaw_profile(np.zeros((1, 3, 5)), *args)


def test_continuous_heading_feedback_crosses_antipodal_direction():
    native_module()
    coefficients = np.zeros((1, 3, 6))
    coefficients[0, 0, 4] = -1.0
    coefficients[0, 1, 3:5] = (-1.5, 0.1)
    # v=(-1, .1-3t, 0), a=(0, -3, 0), j=0: finite constant thrust axis.
    # The former shortest-angle feedback jumps at pi on this physical curve.
    trajectory = PathCoverTrajectory(coefficients, np.array([0.2]), body_yaw_rate_max=1.5)
    sample = trajectory.sample(np.linspace(0.0, 0.2, 4001))
    heading = np.arctan2(sample.velocity[:, 1], sample.velocity[:, 0])
    wrapped = np.arctan2(np.sin(heading - sample.yaw), np.cos(heading - sample.yaw))
    assert np.count_nonzero(np.abs(np.diff(wrapped)) > np.pi) == 1
    assert np.isfinite(sample.yaw).all()
    z = np.array([0.0, -3.0, 9.81])
    z /= np.linalg.norm(z)
    c = z[0] * np.cos(sample.yaw) + z[1] * np.sin(sample.yaw)
    body_r = z[2] * sample.yaw_rate / (1.0 - c * c)
    assert np.max(np.abs(body_r)) <= 1.5 + 1e-4


def test_yaw_failure_retains_native_candidate_and_serializes_only_on_request():
    native_module()
    coefficients = np.zeros((1, 3, 6))
    coefficients[0, 2, 3] = -9.81

    class Candidate:
        coefficients_reads = 0
        corridor_reads = 0
        durations = np.array([0.2])
        local_goal = np.array([1.0, 2.0, 3.0])
        corridor_indices = np.array([0])
        constraint_metrics = np.arange(6, dtype=float)
        optimizer_status = 1
        cost = 123.0
        time_initialization = {"status": 0, "evaluations": 4, "scale": 1.0, "elapsed_ms": 0.1}
        initial_boundary = {
            "mode": "fixed_pva", "measured_acceleration": [0.0, 0.0, -19.62],
            "reference_acceleration": [0.0, 0.0, -19.62], "thrust_axis": None,
            "force_lower_n": 2.0, "force_upper_n": 20.0,
            "force_guess_raw_n": 7.3575, "force_guess_n": 7.3575,
            "force_n": 7.3575, "force_guess_was_projected": False,
        }

        @property
        def coefficients(self):
            self.coefficients_reads += 1
            return list(coefficients)

        @property
        def corridors(self):
            self.corridor_reads += 1
            return [np.array([[0.0, 0.0, 1.0, -4.0]])]

    candidate = Candidate()
    native = SimpleNamespace(plan=lambda *args: candidate, reset=lambda: None, failed_problem=lambda: None)
    frontend = NativePathCoverFrontend.__new__(NativePathCoverFrontend)
    frontend.config = PathCoverConfig(use_cuda=False)
    frontend._native = native
    frontend._yaw_failure = None
    state = PathCoverState(np.array([1.0, 0.0, 1.0]), np.array([0.1, 0.2, 0.3]),
                           np.array([0.0, 0.0, -19.62]), yaw=0.4, time=5.0)
    with pytest.raises(PathCoverPlanningError, match="upright"):
        frontend.plan(None, state, candidate.local_goal)
    assert frontend._yaw_failure[0] is candidate
    assert candidate.coefficients_reads == 1  # Existing trajectory-construction read only.
    assert candidate.corridor_reads == 0
    problem = frontend.failed_problem()
    assert candidate.coefficients_reads == 2
    assert candidate.corridor_reads == 1
    assert problem["stage"] == "yaw"
    assert problem["initial_yaw"] == 0.4
    assert problem["start_time"] == 5.0
    np.testing.assert_array_equal(problem["initial_pva"],
                                  np.column_stack((state.position, state.velocity, state.acceleration)))
    np.testing.assert_array_equal(problem["candidate"]["coefficients"], coefficients)
    np.testing.assert_array_equal(problem["candidate"]["durations"], candidate.durations)
    assert problem["candidate"]["objective"] == 123.0
    frontend.reset()
    assert frontend.failed_problem() is None

    def software_failure(*args):
        raise RuntimeError("ABI mismatch")

    native.plan = software_failure
    with pytest.raises(RuntimeError, match="ABI mismatch"):
        frontend.plan(None, state, candidate.local_goal)
    assert frontend._yaw_failure is None
