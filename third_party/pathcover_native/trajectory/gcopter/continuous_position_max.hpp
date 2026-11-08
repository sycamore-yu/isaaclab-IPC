#pragma once
#include <Eigen/Eigen>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include "root_finder.hpp"

namespace pathcover_continuous_position {
struct Maximum {
    double value = -std::numeric_limits<double>::infinity();
    double alpha = 0.0;
    int face = -1;
    bool finite = true;
};
inline double valueAt(const std::array<double, 6> &a, double alpha) {
    double value = 0.0, power = 1.0;
    for (int k = 0; k < 6; ++k) {
        value += a[k] * power;
        power *= alpha;
    }
    return value;
}
// Upper bound for every floating-point valueAt(a, alpha), 0 <= alpha <= 1.
// Bernstein b_i = sum_{k<=i} choose(i,k)/choose(5,k) * a_k.
// The exact polynomial is bounded by max(b_i). The guard covers the positive
// weights' conversion, <=6-term control sums, valueAt's powers/products/sums,
// and gradual-underflow absolute errors. Uncertain/overflowing bounds do not
// prune. This bounds evaluations of the same rounded coefficients; it does not
// certify completeness of the analytic quartic root solver.
// Assumes IEEE binary64 round-to-nearest and gradual underflow, without
// unsafe fast-math. Near ties or an unusable bound retain the root computation.
inline double bernsteinEvaluationUpperBound(const std::array<double, 6> &a) {
    constexpr double epsilon = std::numeric_limits<double>::epsilon();
    constexpr double tiny = std::numeric_limits<double>::denorm_min();
    constexpr double gamma64 = (64.0 * epsilon) / (1.0 - 64.0 * epsilon);
    constexpr double choose[6][6] = {
        {1,0,0,0,0,0}, {1,1,0,0,0,0}, {1,2,1,0,0,0},
        {1,3,3,1,0,0}, {1,4,6,4,1,0}, {1,5,10,10,5,1}};
    constexpr double degreeChoose[6] = {1,5,10,10,5,1};
    const double infinity = std::numeric_limits<double>::infinity();
    double magnitude = 0.0;
    for (const double value : a) magnitude += std::abs(value);
    const double magnitudeUpper = std::nextafter(
        (magnitude + 16.0 * tiny) / (1.0 - 8.0 * epsilon), infinity);
    if (!std::isfinite(magnitudeUpper)) return infinity;
    double largestControl = -infinity;
    for (int i = 0; i < 6; ++i) {
        double control = a[0];
        for (int k = 1; k <= i; ++k)
            control += (choose[i][k] / degreeChoose[k]) * a[k];
        if (!std::isfinite(control)) return infinity;
        largestControl = std::max(largestControl, control);
    }
    const double guard = std::nextafter(
        gamma64 * magnitudeUpper + 128.0 * tiny, infinity);
    // Preserve nonfinite behavior of intermediate original power sums, even
    // when Bernstein control coefficients strongly cancel.
    if (!std::isfinite(magnitudeUpper + guard)) return infinity;
    return std::nextafter(largestControl + guard, infinity);
}

// Fixed degree, normalized alpha. An unpruned face evaluates both endpoints
// separately; a strictly dominated whole face skips all its candidate times.
// RootFinder's <=quartic route is analytic; its tol argument does not bound
// the resulting error. This is a floating-point extremum, not an interval proof.
// The penalty-only instantiation may omit strictly negative faces. If every
// face is omitted, face == -1 denotes an inactive penalty; signed diagnostics
// retain the default instantiation and its original negative maximum.
template <bool PositivePenaltyOnly = false>
inline Maximum maximum(const Eigen::Matrix<double, 6, 3> &c, double duration,
                       const Eigen::MatrixX4d &planes) {
    Maximum best;
    if constexpr (PositivePenaltyOnly) best.value = 0.0;
    if (!(duration > 0.0) || !std::isfinite(duration) || !c.allFinite()
        || planes.rows() == 0 || !planes.allFinite()) {
        best.finite = false;
        best.value = std::numeric_limits<double>::quiet_NaN();
        return best;
    }
    std::array<double, 6> powers;
    powers[0] = 1.0;
    for (int k = 1; k < 6; ++k) powers[k] = powers[k - 1] * duration;
    for (int face = 0; face < planes.rows(); ++face) {
        std::array<double, 6> a;
        for (int k = 0; k < 6; ++k) {
            a[k] = ((planes(face, 0) * c(k, 0) + planes(face, 1) * c(k, 1))
                     + planes(face, 2) * c(k, 2)) * powers[k];
        }
        a[0] += planes(face, 3);
        Eigen::Matrix<double, 5, 1> derivativeCoefficients;
        for (int k = 0; k < 5; ++k)
            derivativeCoefficients(k) = (5 - k) * a[5 - k];
        const auto take = [&](double alpha) {
            const double value = valueAt(a, alpha);
            if (!std::isfinite(value)) {
                best.finite = false;
                best.value = std::numeric_limits<double>::quiet_NaN();
            } else if (best.finite && (best.face < 0 || value > best.value)) {
                best.value = value;
                best.alpha = alpha;
                best.face = face;
            }
        };
        bool coefficients_finite = derivativeCoefficients.allFinite();
        for (const double value : a) coefficients_finite &= std::isfinite(value);
        if (!coefficients_finite) {
            best.finite = false;
            best.value = std::numeric_limits<double>::quiet_NaN();
            return best;
        }
        // Strict separation retains zero/positive first-face/first-alpha ties.
        // Every face still passes the original finite/overflow checks first.
        const double threshold = PositivePenaltyOnly ? std::max(0.0, best.value) : best.value;
        if (best.finite && (PositivePenaltyOnly || best.face >= 0) &&
            bernsteinEvaluationUpperBound(a) < threshold) {
            continue;
        }
        take(0.0);
        const double scale = derivativeCoefficients.cwiseAbs().maxCoeff();
        if (scale != 0.0) {
            Eigen::VectorXd derivative = derivativeCoefficients;
            derivative /= scale;
            const auto roots = RootFinder::solvePolynomial(derivative, 0.0, 1.0, 1e-12);
            for (const double alpha : roots) take(alpha);
        }
        take(1.0);
        if (!best.finite) {
            return best;
        }
    }
    return best;
}
} // namespace pathcover_continuous_position

