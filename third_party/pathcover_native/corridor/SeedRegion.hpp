#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace PathCover {

class SeedRegionCollision : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// The triangle/box separating-axis family is complete, including degenerate
// triangles. Point-to-box normals improve the cut without weakening separation.
// The caller supplies the three world points; this geometry does not infer a
// dynamics model, control period, or a guaranteed feasible future trajectory.
template<typename T>
inline std::pair<Eigen::Matrix<T, 3, 1>, T> seedHullPlane(
    const std::vector<Eigen::Matrix<T, 3, 1>> &points,
    const Eigen::Matrix<T, 3, 1> &centre,
    const Eigen::Matrix<T, 3, 1> &half, T alpha) {
    using V = Eigen::Matrix<T, 3, 1>;
    if (points.size() != 3) {
        throw std::invalid_argument("Initial seed hull needs three points");
    }
    std::vector<V> axes;
    axes.reserve(16);
    const std::array<V, 3> basis{V::UnitX(), V::UnitY(), V::UnitZ()};
    for (const auto &axis : basis) axes.push_back(axis);
    const V edges[3] = {points[1] - points[0], points[2] - points[1], points[0] - points[2]};
    axes.push_back(edges[0].cross(edges[1]));
    for (const auto &edge : edges) {
        for (const auto &axis : basis) axes.push_back(edge.cross(axis));
    }
    for (const auto &point : points) {
        const V delta = centre - point;
        axes.push_back((delta.array().sign() * (delta.array().abs() - half.array()).max(T(0))).matrix());
    }
    T best_gap = -std::numeric_limits<T>::infinity();
    T best_support = T(0);
    V best = V::Zero();
    for (const auto &candidate : axes) {
        const T length = candidate.norm();
        if (length <= T(1e-12)) continue;
        for (T sign : {T(-1), T(1)}) {
            const V normal = candidate * (sign / length);
            T support = -std::numeric_limits<T>::infinity();
            for (const auto &point : points) support = std::max(support, normal.dot(point));
            const T near = normal.dot(centre) - normal.cwiseAbs().dot(half);
            if (near - support > best_gap) {
                best_gap = near - support;
                best_support = support;
                best = normal;
            }
        }
    }
    if (best_gap <= T(1e-10)) {
        throw SeedRegionCollision("PathCover initial motion seed intersects an occupied voxel");
    }
    return {best, best_support + (T(1) - alpha) * best_gap};
}

template<typename T>
inline std::pair<Eigen::Matrix<T, 3, 1>, T> seedHullPlane(
    const std::vector<Eigen::Matrix<T, 3, 1>> &points,
    const Eigen::Matrix<T, 3, 1> &centre, T half, T alpha) {
    const Eigen::Matrix<T, 3, 1> widths = Eigen::Matrix<T, 3, 1>::Constant(half);
    return seedHullPlane(points, centre, widths, alpha);
}

} // namespace PathCover
