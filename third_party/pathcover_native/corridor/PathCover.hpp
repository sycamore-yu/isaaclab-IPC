#ifndef PATHCOVER_HPP
#define PATHCOVER_HPP

#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <algorithm>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <libqhullcpp/Qhull.h>
#include "SeedRegion.hpp"


namespace PathCover {

// Per-thread RNG: independent planner calls do not race on the C global rand().
inline thread_local std::mt19937 random_engine(0);
inline void setRandomSeed(unsigned int seed) { random_engine.seed(seed); }


template<typename T, const int dim> // Uses qhull
inline void qConvexHull(Eigen::Matrix<T, -1, -1, Eigen::RowMajor> &Points, 
                        std::vector<int> &idx) {
    
    std::vector<double> points_mapped(Points.data(), Points.data() + Points.size());
    orgQhull::Qhull qhull = orgQhull::Qhull("", dim, Points.rows(), points_mapped.data(), "");
    
    idx.reserve(qhull.vertexList().size());
    for (auto vi = qhull.vertexList().begin(); vi != qhull.vertexList().end(); ++vi) {
      idx.push_back(vi->point().id());
    }
}    


template<typename T, const int dim>
inline void noredund(Eigen::Matrix<T, -1, dim> &A, 
                     Eigen::Matrix<T, -1, 1> &b, 
                     const Eigen::Matrix<T, dim, 1> &seed) {

    // Ensure seed is strictly inside the polytope by checking A*seed < b
    Eigen::Matrix<T, -1, 1> check = b - A * seed;
    for (int i = 0; i < check.size(); ++i) {
        if (check(i) <= 1e-8) {  // Allow small tolerance
            std::cout << A << "\n" << b.transpose() << "\n" << seed.transpose() << "\n";
            throw std::invalid_argument("Seed point is not strictly inside the polytope. Redundancy removal failed!");
        }
    }                        
    
    // obtain dual polytope vertices
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> D = (A.array().colwise() / (b - A * seed).array());
    

    std::vector<int> idx;

    qConvexHull<T, dim>(D, idx);

    
    // Temporary matrices to store the rows corresponding to the convex hull
    Eigen::Matrix<T, -1, dim> An(idx.size(), A.cols());
    Eigen::Matrix<T, -1, 1> bn(idx.size());

    for (int i = 0; i < idx.size(); ++i) {
        An.row(i) = A.row(idx[i]);
        bn(i) = b(idx[i]);
    }

    A = An;
    b = bn;
}


// RISP (Randomized Iterative Space Partitioning), Algorithm 2 of the paper.
// For each randomly sampled obstacle point p, the cutting hyperplane is
//
//     a = p - y_seed,
//     b = a^T y_seed + (1 - alpha) * ||a||^2,
//
// which is Eq. (4) of the paper. The plane passes through the point
// alpha*y_seed + (1-alpha)*p and the open ball of radius (1-alpha)*||a||
// around y_seed is strictly contained in the half-space a^T x <= b.
// alpha must lie in (0, 1); smaller alpha -> larger polytope but less margin,
// larger alpha -> smaller polytope with a wider safety annulus around the seed.
template<typename T, const int dim>
inline void RISP(const std::vector<Eigen::Matrix<T, dim, 1>> &obstacle_pts, 
                 const Eigen::Matrix<T, dim, 1> &seed, 
                 const Eigen::Matrix<T, -1, dim> &A_bound, 
                 const Eigen::Matrix<T, -1, 1> &b_bound, 
                 Eigen::Matrix<T, -1, dim> &A, 
                 Eigen::Matrix<T, -1, 1> &b,
                 std::vector<int> &pts_remaining,
                 T alpha = T(0.01),
                 T voxel_half_extent = T(0),
                 const std::vector<Eigen::Matrix<T, dim, 1>> *seed_region = nullptr) {
    assert(alpha > T(0) && alpha < T(1) && "RISP: alpha must lie in (0, 1)");

    // Compact indices instead of copying and moving every XYZ point on each cut.
    std::vector<std::size_t> remaining(obstacle_pts.size());
    std::iota(remaining.begin(), remaining.end(), 0);
    std::vector<Eigen::Matrix<T, dim, 1>> normals;
    std::vector<T> offsets;
    normals.reserve(64);
    offsets.reserve(64);
    pts_remaining.clear();
    pts_remaining.push_back(static_cast<int>(remaining.size()));
    const bool protect_region = seed_region && std::any_of(seed_region->begin(), seed_region->end(),
        [&](const auto &point) { return (point - seed).squaredNorm() > T(1e-24); });
    while (!remaining.empty()) {
        std::uniform_int_distribution<std::size_t> sample(0, remaining.size() - 1);
        const auto &centre = obstacle_pts[remaining[sample(random_engine)]];
        const auto delta = (centre - seed).eval();
        // The closest point on an occupied AABB gives a separating normal.
        // A point obstacle is the zero-half-extent special case of the same cut.
        Eigen::Matrix<T, dim, 1> normal =
            (delta.array().sign() * (delta.array().abs() - voxel_half_extent).max(T(0))).matrix();
        if (normal.squaredNorm() == T(0)) {
            throw std::invalid_argument("RISP seed lies in an occupied voxel volume");
        }
        T offset = normal.dot(seed) + (T(1) - alpha) * normal.squaredNorm();
        if constexpr (dim == 3) {
            if (protect_region) {
                const auto plane = seedHullPlane(*seed_region, centre, voxel_half_extent, alpha);
                normal = plane.first;
                offset = plane.second;
            }
        }
        normals.push_back(normal);
        offsets.push_back(offset);
        const T support = voxel_half_extent * normal.template lpNorm<1>();
        remaining.erase(std::remove_if(remaining.begin(), remaining.end(),
            [&](std::size_t index) { return normal.dot(obstacle_pts[index]) - support > offset; }),
            remaining.end());
        pts_remaining.push_back(static_cast<int>(remaining.size()));
    }
    const int count = static_cast<int>(normals.size());
    A.resize(count + A_bound.rows(), dim);
    b.resize(count + b_bound.size());
    for (int i = 0; i < count; ++i) {
        A.row(i) = normals[i].transpose();
        b(i) = offsets[i];
    }
    A.bottomRows(A_bound.rows()) = A_bound;
    b.tail(b_bound.size()) = b_bound;

    // Keep each voxel's separating cut as an explicit certificate. Dropping a
    // redundant face can replace it with a combination of faces, which prevents
    // the direct one-face-per-box separation check used at the grid/SFC boundary.
    if (voxel_half_extent == T(0)) noredund<T, dim>(A, b, seed);
}


template <typename T, int Dim>
inline Eigen::Matrix<T, Dim, 1> computePolyhedraIntersection(const Eigen::Matrix<T, -1, Dim> &A, 
                                                             const Eigen::Matrix<T, -1, 1> &b,
                                                             const Eigen::Matrix<T, Dim, 1> &p1, 
                                                             const Eigen::Matrix<T, Dim, 1> &p2) {
    T t_min = std::numeric_limits<T>::infinity();
    const Eigen::Matrix<T, Dim, 1> direction = p2 - p1;
    for (int i = 0; i < A.rows(); ++i) {
        const T denom = A.row(i).dot(direction);
        // Only an outward crossing can leave a convex region containing p1.
        if (denom <= T(1e-12)) continue;
        const T t = (b(i) - A.row(i).dot(p1)) / denom;
        if (t >= T(-1e-9) && t <= T(1) + T(1e-9)) {
            t_min = std::min(t_min, std::clamp(t, T(0), T(1)));
        }
    }
    if (!std::isfinite(t_min)) {
        throw std::runtime_error("PathCover: path segment has no exit intersection");
    }
    return p1 + t_min * direction;
}


template <typename T, int Dim>
inline void pathCover(const std::vector<Eigen::Matrix<T, Dim, 1>> &obstacle_pts, 
                      const std::vector<Eigen::Matrix<T, Dim, 1>> &path,
                      const Eigen::Matrix<T, -1, Dim> &A_bound, 
                      const Eigen::Matrix<T, -1, 1> &b_bound,
                      std::vector<Eigen::Matrix<T, -1, Dim>> &A, 
                      std::vector<Eigen::Matrix<T, -1, 1>> &b,
                      std::vector<Eigen::Matrix<T, Dim, 1>> &seed, 
                      std::vector<std::vector<int>> &all_pts_remaining,
                      int max_horizon = 1,
                      T alpha = T(0.01),
                      T voxel_half_extent = T(0),
                      const std::vector<Eigen::Matrix<T, Dim, 1>> &initial_seed_region = {}) {

    if (path.empty() || max_horizon < 1) {
        throw std::invalid_argument("PathCover needs a path and a positive corridor horizon");
    }
    A.clear();
    b.clear();
    seed.clear();
    all_pts_remaining.clear();
    Eigen::Matrix<T, -1, Dim> current_A;
    Eigen::Matrix<T, -1, 1> current_b;
    std::vector<int> decay;
    auto append = [&](const Eigen::Matrix<T, Dim, 1> &point) {
        RISP<T, Dim>(obstacle_pts, point, A_bound, b_bound,
                     current_A, current_b, decay, alpha, voxel_half_extent,
                     A.empty() && !initial_seed_region.empty() ? &initial_seed_region : nullptr);
        A.push_back(current_A);
        b.push_back(current_b);
        seed.push_back(point);
        all_pts_remaining.push_back(decay);
    };
    Eigen::Matrix<T, Dim, 1> last = path.front();
    append(last);
    for (std::size_t i = 1; i < path.size();) {
        const auto residual = (current_A * path[i] - current_b).eval();
        if ((residual.array() <= T(1e-9) * current_A.rowwise().norm().array()).all()) {
            last = path[i++];
            continue;
        }
        last = computePolyhedraIntersection<T, Dim>(current_A, current_b, last, path[i]);
        if (static_cast<int>(A.size()) == max_horizon) break;
        append(last);
    }
    // One seed per corridor, followed by the endpoint covered by this horizon.
    seed.push_back(last);
}

}

#endif // PATHCOVER_HPP
