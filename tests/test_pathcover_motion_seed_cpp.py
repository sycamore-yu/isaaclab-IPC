"""One-dimensional motion seed projection against independent HiGHS LP oracles."""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import linprog

SOURCE = r"""
#include "pathcover_motion_seed.h"
#include "corridor/SeedRegion.hpp"
#include <iomanip>
#include <iostream>
using namespace pathcover_motion_seed;
int main() {
    std::cout << std::setprecision(17);
    char mode;
    while (std::cin >> mode) {
        Triangle seed; TipSegment tips; Point lower, upper; int count;
        for (auto& p : seed) for (int i=0;i<3;++i) std::cin >> p[i];
        for (auto& p : tips) for (int i=0;i<3;++i) std::cin >> p[i];
        for (int i=0;i<3;++i) std::cin >> lower[i];
        for (int i=0;i<3;++i) std::cin >> upper[i];
        std::cin >> count; std::vector<Box> boxes(count);
        for (auto& box : boxes) {
            for (int i=0;i<3;++i) std::cin >> box.lower[i];
            for (int i=0;i<3;++i) std::cin >> box.upper[i];
        }
        sdlp::permutation_rng.seed(0);
        try {
            if (mode=='I') {
                const auto interval=collisionInterval(seed[0],seed[1],tips,boxes.at(0),{0.,1.});
                if (interval) std::cout << "INTERVAL " << interval->lower << ' ' << interval->upper << '\n';
                else std::cout << "EMPTY\n";
            } else {
                const auto result=select(seed,tips,boxes,lower,upper);
                // Independently call the production corridor separator too.
                const std::vector<Point> points(result.points.begin(),result.points.end());
                for (const auto& box : boxes) PathCover::seedHullPlane<double>(
                    points,(box.lower+box.upper)*.5,Point((box.upper-box.lower)*.5),.01);
                std::cout << "OK " << result.alpha << ' ' << result.adjusted;
                for (const auto& point : result.points) for (int i=0;i<3;++i) std::cout << ' ' << point[i];
                std::cout << '\n';
            }
        } catch (const NoFeasibleSeed& error) { std::cout << "NONE " << error.what() << '\n'; }
          catch (const std::exception& error) { std::cout << "ERROR " << error.what() << '\n'; }
    }
}
"""


@pytest.fixture(scope="module")
def runner(tmp_path_factory):
    compiler = shutil.which("g++")
    if not compiler or not Path("/usr/include/eigen3/Eigen/Core").exists():
        pytest.skip("C++ geometry tests require g++ and Eigen")
    root = Path(__file__).resolve().parents[1]
    vendor = root / "third_party/pathcover_native"
    build = tmp_path_factory.mktemp("motion_seed")
    source, binary = build / "main.cpp", build / "motion_seed"
    source.write_text(SOURCE)
    subprocess.run(
        [compiler, "-std=c++17", "-O2", "-I/usr/include/eigen3", f"-I{root / 'native'}",
         f"-I{vendor}", f"-I{vendor / 'trajectory'}", f"-I{vendor / 'occupancy_inflation/include'}",
         str(source), "-o", str(binary)], check=True, capture_output=True, text=True,
    )

    def run(cases):
        lines = []
        for mode, seed, tips, boxes, bounds in cases:
            values = np.r_[np.asarray(seed).ravel(), np.asarray(tips).ravel(),
                           np.asarray(bounds).ravel(), len(boxes), np.asarray(boxes).ravel()]
            lines.append(mode + " " + " ".join(format(float(value), ".17g") for value in values))
        result = subprocess.run([str(binary)], input="\n".join(lines) + "\n",
                                capture_output=True, check=True, text=True)
        return [line.split() for line in result.stdout.splitlines()]

    return run


BOUNDS = np.array([[-20.0] * 3, [20.0] * 3])


def oracle_interval(seed, tips, box):
    """Independent scipy/HiGHS projection, without the vendored Seidel solver."""
    p0, p1 = np.asarray(seed)[:2]
    a, tip_end = np.asarray(tips)
    lower, upper = np.asarray(box)
    d, e, f = p1 - p0, a - p0, tip_end - a
    matrix = np.vstack((np.column_stack((d, p0 - upper, f)),
                        np.column_stack((-d, lower - p0, -f)), [1.0, -1.0, 0.0]))
    rhs = np.r_[-e, e, -1.0]
    answers = [linprog([0.0, 0.0, sign], A_ub=matrix, b_ub=rhs,
                       bounds=[(0.0, None), (1.0, None), (0.0, 1.0)], method="highs")
               for sign in (1.0, -1.0)]
    if answers[0].status == 2:
        assert answers[1].status == 2
        return None
    assert all(answer.success for answer in answers)
    return np.array([answers[0].x[2], answers[1].x[2]])


def test_box_intervals_match_independent_lp_including_degenerate_geometry(runner):
    rng = np.random.default_rng(12025)
    cases = []
    for i in range(72):
        p0 = np.array([-2.0, 0.0, 0.0])
        p1 = p0.copy() if i % 4 == 0 else np.array([-1.5, 0.2, 0.0])
        tips = rng.uniform(-1.0, 1.0, (2, 3)) + [1.0, 0.0, 0.0]
        centre = rng.uniform(-0.5, 0.5, 3)
        half = rng.uniform(0.05, 0.4, 3)
        if i % 3 == 0:
            tips[:, 2] = 0.0
            centre[2] = half[2]  # An exactly coplanar box face.
        seed = np.array([p0, p1, tips.mean(axis=0)])
        cases.append(("I", seed, tips, [[centre - half, centre + half]], BOUNDS))
    for case, result in zip(cases, runner(cases), strict=True):
        expected = oracle_interval(case[1], case[2], case[3][0])
        if expected is None:
            assert result == ["EMPTY"]
        else:
            assert result[0] == "INTERVAL", result
            np.testing.assert_allclose(np.asarray(result[1:], float), expected, rtol=0.0, atol=2e-8)


def test_empty_map_and_safe_warm_seed_preserve_original_bits(runner):
    tips = np.array([[0.7, 0.11, 0.2], [0.7, 0.11, 1.2]])
    seed = np.array([[-1.0, 0.0, 0.2], [-0.5, 0.1, 0.2], tips[0] + 0.37 * (tips[1] - tips[0])])
    cases = [("S", seed, tips, boxes, BOUNDS) for boxes in ([], [[[2, 2, 2], [3, 3, 3]]])]
    for result in runner(cases):
        assert result[0] == "OK" and result[2] == "0"
        np.testing.assert_array_equal(np.asarray(result[3:], float).reshape(3, 3), seed)


def test_fixed_edge_or_entire_tip_segment_collision_remains_failure(runner):
    tips = np.array([[1, -0.2, 0], [1, 0.2, 0]], dtype=float)
    seed = np.array([[-1, 0, 0], [0, 0, 0], tips.mean(axis=0)])
    boxes = [[[[-0.6, -0.1, -0.1], [-0.4, 0.1, 0.1]]],
             [[[0.8, -0.3, -0.1], [1.2, 0.3, 0.1]]]]
    answers = runner([("S", seed, tips, box, BOUNDS) for box in boxes])
    assert all(answer[0] == "NONE" for answer in answers)
    assert "fixed" in answers[0]


def test_multiple_free_components_touch_and_narrow_gap(runner):
    p0 = np.array([0.0, 0.0, 0.0])
    tips = np.array([[2.0, -2.0, 0.0], [2.0, 2.0, 0.0]])
    # Two blockers yield separate free angular components. Degenerate p0=p1
    # tests the segment case without requiring an invertible tetrahedron.
    boxes = np.array([[[1, -0.8, -0.1], [1.1, -0.4, 0.1]],
                      [[1, 0.4, -0.1], [1.1, 0.8, 0.1]]])
    seed = np.array([p0, p0, [2.0, 1.0, 0.0]])
    result = runner([("S", seed, tips, boxes, BOUNDS)])[0]
    assert result[0] == "OK" and result[2] == "1", result
    # Closest free component is the central gap, at alpha just below 0.6818...
    intervals = [oracle_interval(seed, tips, box) for box in boxes]
    alpha = float(result[1])
    assert max(intervals[0]) < alpha < min(intervals[1])
    assert min(intervals[1]) - alpha < 3e-8

    gap = 1e-6
    narrow_boxes = np.array([[[1, -2, -0.1], [1.1, 0, 0.1]],
                            [[1, gap, -0.1], [1.1, 2, 0.1]]])
    touching_seed = np.array([p0, p0, [2.0, 0.0, 0.0]])
    result = runner([("S", touching_seed, tips, narrow_boxes, BOUNDS)])[0]
    assert result[0] == "OK" and result[2] == "1", result
    assert 0.5 < float(result[1]) < 0.5 + gap / 2.2


def test_workspace_clips_tip_range_without_moving_fixed_points(runner):
    tips = np.array([[1.0, 0.0, -1.0], [1.0, 0.0, 2.0]])
    seed = np.array([[0.0, 0.0, 0.5], [0.2, 0.0, 0.5], tips[0]])
    bounds = np.array([[-2.0, -2.0, 0.0], [2.0, 2.0, 1.0]])
    result = runner([("S", seed, tips, [], bounds)])[0]
    assert result[0] == "OK" and result[2] == "1", result
    selected = np.asarray(result[3:], float).reshape(3, 3)
    np.testing.assert_array_equal(selected[:2], seed[:2])
    assert 0.0 <= selected[2, 2] <= 1e-7


def test_dense_v24_force_interval_and_selected_triangle(runner):
    # Frozen input SHA5d2f0dee...7ff. These are logical fine AABBs, not the
    # packed-float diagnostic centres. The unchanging base is physically free.
    p0 = np.array([-2.6578736305236816, 45.85956954956055, 0.3574524521827698])
    p1 = np.array([-2.6448037829850306, 46.02970105601192, 0.3315343608756091])
    base = np.array([-2.6317339354463796, 46.19983256246329, 0.20751626956844843])
    slope = np.array([0.00159749456768048, 0.0010849558735933852, 0.01319275047731031])
    seed = np.array([p0, p1, [-2.6098161551344083, 46.21471826221258, 0.38852208487596984]])
    tips = np.array([base, base + 22.0725 * slope])
    indices = np.array([[-27, 458, 2], [-26, 458, 2], [-27, 459, 2], [-26, 459, 2],
                        [-27, 460, 2], [-26, 460, 2], [-27, 461, 2], [-26, 461, 2],
                        [-27, 462, 2], [-26, 462, 2], [-27, 462, 3], [-26, 462, 3]])
    boxes = np.stack((indices * 0.1, (indices + 1) * 0.1), axis=1)
    bounds = np.array([[-7.3, -54.8, 0.2], [7.3, 54.8, 3.3]])
    result = runner([("S", seed, tips, boxes, bounds)])[0]
    assert result[0] == "OK" and result[2] == "1", result
    force = float(result[1]) * 22.0725
    assert 15.0837187899 < force < 15.08372
    np.testing.assert_array_equal(np.asarray(result[3:], float).reshape(3, 3)[:2], seed[:2])
