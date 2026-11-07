"""Preserve the scalar plane dot order through the real 3D penalty caller."""

import shutil
import subprocess
from pathlib import Path

import pytest

SOURCE = r"""
#include <math.h>
#include <algorithm>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include "gcopter_solver.hpp"

using Solver = trajectory_server::GcopterSolver<3>;

void require(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
}
bool same(double a, double b) { return std::memcmp(&a, &b, sizeof(a)) == 0; }

void check(int faces) {
    std::mt19937_64 rng(327 + faces);
    std::uniform_real_distribution<double> number(-1., 1.);
    Solver::PolyhedraH raw(2), inner(2);
    Eigen::MatrixX3d coefficients = Eigen::MatrixX3d::Zero(12, 3);
    Eigen::VectorXd times(2), bounds(5), weights(5);
    Eigen::VectorXi indices(2);
    times << .3, .2; indices << 0, 1;
    bounds << 10., 10., 3., 0., 100.;
    weights << 1e5, 100., 1e4, 1e4, 1e4;
    for (int piece = 0; piece < 2; ++piece) {
        const int count = faces + piece;
        raw[piece].resize(count, 4);
        inner[piece].resize(count, 4);
        const Eigen::Vector3d pos(1., 1., 1.);
        coefficients.row(6 * piece) = pos;
        for (int k = 0; k < count; ++k) {
            Eigen::Vector3d normal(number(rng), number(rng), number(rng));
            // Reassociation changes this first projection from 1 to 0.
            if (k == 0) normal << 1e16, -1e16, 1.;
            const double offset = -normal.dot(pos) + (k % 3 == 0 ? .05 : -.2);
            raw[piece].row(k).head<3>() = normal;
            raw[piece](k, 3) = offset;
            inner[piece].row(k) = raw[piece].row(k);
            inner[piece](k, 3) += k % 2 == 0 ? .01 : .02;
        }
    }
    flatness::FlatnessMap map;
    map.reset(.75, 9.81, 0., 0., 0., .001);
    Solver::ConstraintMetrics metrics;
    Eigen::VectorXd gradT = Eigen::VectorXd::Zero(2);
    Eigen::MatrixX3d gradC = Eigen::MatrixX3d::Zero(12, 3);
    double cost = 0.;
    Solver::attachPenaltyFunctional<true>(times, coefficients, indices, inner, .01, 64,
        bounds, weights, map, metrics, 1e-6, raw, cost, gradT, gradC);

    double expectedCost = 0., expectedSfc = -std::numeric_limits<double>::infinity();
    Eigen::VectorXd expectedT = Eigen::VectorXd::Zero(2);
    Eigen::MatrixX3d expectedC = Eigen::MatrixX3d::Zero(12, 3);
    const Eigen::Vector3d zero = Eigen::Vector3d::Zero();
    const double frac = 1. / 64.;
    for (int i = 0; i < 2; ++i) {
        const double step = times[i] * frac;
        const Eigen::Matrix<double, 6, 3> c = coefficients.block<6, 3>(6 * i, 0);
        for (int j = 0; j <= 64; ++j) {
            Eigen::Matrix<double, 6, 1> b0, b1, b2, b3, b4;
            Solver::fillDerivativeBasis<false>(j * step, b0, b1, b2, b3, b4);
            const Eigen::Vector3d pos = c.transpose() * b0;
            Eigen::Vector3d gradient = Eigen::Vector3d::Zero();
            double penalty = 0.;
            for (int k = 0; k < inner[i].rows(); ++k) {
                const Eigen::Vector3d normal = inner[i].block<1, 3>(k, 0);
                const double projection = normal.dot(pos);
                expectedSfc = std::max(expectedSfc, projection + raw[i](k, 3));
                double value, derivative;
                if (Solver::smoothedL1(projection + inner[i](k, 3), 1e-6, value, derivative)) {
                    gradient += weights[0] * derivative * normal;
                    penalty += weights[0] * value;
                }
            }
            const double node = (j == 0 || j == 64) ? .5 : 1.;
            expectedC.block<6, 3>(6 * i, 0) +=
                (b0 * gradient.transpose() + b1 * zero.transpose()
                 + b2 * zero.transpose() + b3 * zero.transpose()) * node * step;
            expectedT[i] += node * frac * penalty;
            expectedCost += node * step * penalty;
        }
    }
    require(metrics.finite, "finite fixture became invalid");
    require(same(cost, expectedCost), "position penalty changed scalar accumulation");
    require(same(metrics.sfc, expectedSfc), "original SFC maximum changed");
    for (int i = 0; i < gradT.size(); ++i)
        require(same(gradT[i], expectedT[i]), "time gradient changed");
    for (int i = 0; i < gradC.size(); ++i)
        require(same(gradC.data()[i], expectedC.data()[i]), "coefficient gradient changed");
}

int main() {
    for (int faces : {1, 2, 3, 7, 8, 9, 31, 32, 33, 64, 65, 128}) check(faces);
}
"""


def test_plane_projection_preserves_scalar_cost_gradients_and_tail_faces(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("native compiler is unavailable")
    root = Path(__file__).resolve().parents[1]
    vendor = root / "third_party/pathcover_native/trajectory"
    header = (vendor / "gcopter_solver.hpp").read_text()
    pivot = header.index("#elif PATHCOVER_DIM == 3")
    # Expose the actual quadrature caller without changing its arithmetic.
    (tmp_path / "gcopter_solver.hpp").write_text(
        header[:pivot] + header[pivot:].replace("    private:", "    public:")
    )
    source = tmp_path / "contract.cpp"
    binary = tmp_path / "contract"
    source.write_text(SOURCE)
    subprocess.run(
        [compiler, "-O3", "-DNDEBUG", "-std=c++17", "-DPATHCOVER_DIM=3",
         "-I/usr/include/eigen3", f"-I{tmp_path}", f"-I{vendor}", str(source), "-o", str(binary)],
        check=True, capture_output=True, text=True,
    )
    subprocess.run([str(binary)], check=True, capture_output=True, text=True)
