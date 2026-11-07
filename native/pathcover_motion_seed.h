#pragma once

#include <Eigen/Geometry>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

#include "gcopter/sdlp.hpp"
#include "pathcover_endpoint.h"

namespace pathcover_motion_seed {

using Point = Eigen::Vector3d;
using Triangle = std::array<Point, 3>;
using TipSegment = std::array<Point, 2>;

struct Box { Point lower, upper; };
struct Interval { double lower, upper; };
struct Selection {
    Triangle points;
    double alpha = 0.0;
    bool adjusted = false;
};

class NoFeasibleSeed : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// The same complete separating-axis family and numerical gap as seedHullPlane.
// This predicate avoids constructing a corridor or changing occupied geometry.
inline bool separated(const Triangle& points, const Box& box) {
    const Point centre = (box.lower + box.upper) * 0.5;
    const Point half = (box.upper - box.lower) * 0.5;
    const std::array<Point, 3> basis{Point::UnitX(), Point::UnitY(), Point::UnitZ()};
    const Point edges[3]{points[1] - points[0], points[2] - points[1], points[0] - points[2]};
    auto axisSeparates = [&](const Point& axis) {
        const double length = axis.norm();
        if (length <= 1e-12) return false;
        for (double sign : {-1.0, 1.0}) {
            const Point normal = axis * (sign / length);
            double support = -std::numeric_limits<double>::infinity();
            for (const auto& point : points) support = std::max(support, normal.dot(point));
            if (normal.dot(centre) - normal.cwiseAbs().dot(half) - support > 1e-10) return true;
        }
        return false;
    };
    for (const auto& axis : basis) if (axisSeparates(axis)) return true;
    if (axisSeparates(edges[0].cross(edges[1]))) return true;
    for (const auto& edge : edges) for (const auto& axis : basis) {
        if (axisSeparates(edge.cross(axis))) return true;
    }
    for (const auto& point : points) {
        const Point delta = centre - point;
        if (axisSeparates((delta.array().sign()
            * (delta.array().abs() - half.array()).max(0.0)).matrix())) return true;
    }
    return false;
}

// With a fixed free base p0--p1, intersection of triangle(p0,p1,a+alpha*f)
// with a convex box projects to one closed alpha interval. Let a point of the
// triangle be p0+beta*d+gamma*(e+alpha*f). For gamma>0, u=beta/gamma,
// t=1/gamma give a three-variable LP in (u,t,alpha). gamma=0 is the fixed base
// and must be checked before calling this routine. No force sampling is used.
inline std::optional<Interval> collisionInterval(
    const Point& p0, const Point& p1, const TipSegment& tips,
    const Box& box, const Interval& domain) {
    const Point d = p1 - p0, e = tips[0] - p0, f = tips[1] - tips[0];
    Eigen::Matrix<double, 11, 3> all_A;
    Eigen::Matrix<double, 11, 1> all_b;
    for (int axis = 0; axis < 3; ++axis) {
        all_A.row(axis) << d[axis], p0[axis] - box.upper[axis], f[axis];
        all_b[axis] = -e[axis];
        all_A.row(axis + 3) << -d[axis], box.lower[axis] - p0[axis], -f[axis];
        all_b[axis + 3] = e[axis];
    }
    all_A.row(6) << -1.0, 0.0, 0.0; all_b[6] = 0.0;
    all_A.row(7) << 0.0, -1.0, 0.0; all_b[7] = -1.0;
    all_A.row(8) << 1.0, -1.0, 0.0; all_b[8] = -1.0;
    all_A.row(9) << 0.0, 0.0, -1.0; all_b[9] = -domain.lower;
    all_A.row(10) << 0.0, 0.0, 1.0; all_b[10] = domain.upper;
    // Coplanar/point seeds can yield a literal 0 <= 0 box face. The vendor LP
    // normalizes homogeneous rows, so remove only these exact redundancies.
    Eigen::Matrix<double, Eigen::Dynamic, 3> A(11, 3);
    Eigen::VectorXd b(11);
    int count = 0;
    for (int row = 0; row < 11; ++row) {
        if (all_A.row(row).squaredNorm() == 0.0) {
            if (all_b[row] < 0.0) return std::nullopt;
        } else {
            A.row(count) = all_A.row(row);
            b[count++] = all_b[row];
        }
    }
    A.conservativeResize(count, Eigen::NoChange); b.conservativeResize(count);
    Point solution;
    const double lower = sdlp::linprog<3>(Point(0.0, 0.0, 1.0), A, b, solution);
    if (lower == std::numeric_limits<double>::infinity()) return std::nullopt;
    const double upper = -sdlp::linprog<3>(Point(0.0, 0.0, -1.0), A, b, solution);
    if (!std::isfinite(lower) || !std::isfinite(upper) || lower > upper + 1e-10) {
        throw std::runtime_error("Motion seed interval LP failed");
    }
    return Interval{std::clamp(lower, domain.lower, domain.upper),
                    std::clamp(upper, domain.lower, domain.upper)};
}

inline std::vector<Interval> mergeIntervals(std::vector<Interval> intervals) {
    std::sort(intervals.begin(), intervals.end(), [](const Interval& a, const Interval& b) {
        return a.lower < b.lower || (a.lower == b.lower && a.upper < b.upper);
    });
    std::vector<Interval> merged;
    for (const auto& interval : intervals) {
        if (!merged.empty() && interval.lower <= merged.back().upper) {
            merged.back().upper = std::max(merged.back().upper, interval.upper);
        } else merged.push_back(interval);
    }
    return merged;
}

inline Interval workspaceDomain(const Triangle& preferred, const TipSegment& tips,
                                const Point& lower, const Point& upper) {
    for (int i = 0; i < 2; ++i) {
        if ((preferred[i].array() < lower.array()).any()
            || (preferred[i].array() > upper.array()).any()) {
            throw NoFeasibleSeed("PathCover fixed motion seed edge violates workspace body clearance");
        }
    }
    Interval result{0.0, 1.0};
    const Point direction = tips[1] - tips[0];
    for (int axis = 0; axis < 3; ++axis) {
        if (direction[axis] == 0.0) {
            if (tips[0][axis] < lower[axis] || tips[0][axis] > upper[axis]) {
                throw NoFeasibleSeed("PathCover motion tip segment has no point inside workspace body clearance");
            }
        } else {
            double a = (lower[axis] - tips[0][axis]) / direction[axis];
            double b = (upper[axis] - tips[0][axis]) / direction[axis];
            if (a > b) std::swap(a, b);
            result.lower = std::max(result.lower, a);
            result.upper = std::min(result.upper, b);
        }
    }
    if (result.lower > result.upper) {
        throw NoFeasibleSeed("PathCover motion tip segment has no point inside workspace body clearance");
    }
    return result;
}

inline Selection select(const Triangle& preferred, const TipSegment& tips,
                        const std::vector<Box>& boxes, const Point& lower, const Point& upper) {
    for (const auto& point : preferred) if (!point.allFinite()) {
        throw std::invalid_argument("Motion seed points must be finite");
    }
    for (const auto& point : tips) if (!point.allFinite()) {
        throw std::invalid_argument("Motion tip segment points must be finite");
    }
    const Point direction = tips[1] - tips[0];
    const double length2 = direction.squaredNorm();
    const double preference = length2 == 0.0 ? 0.0 : (preferred[2] - tips[0]).dot(direction) / length2;
    if ((preferred[2] - (tips[0] + std::clamp(preference, 0.0, 1.0) * direction)).norm() > 1e-9) {
        throw std::invalid_argument("Preferred motion tip must lie on its supplied segment");
    }
    const Interval domain = workspaceDomain(preferred, tips, lower, upper);
    auto free = [&](const Triangle& points) {
        if ((points[2].array() < lower.array()).any() || (points[2].array() > upper.array()).any()) return false;
        for (const auto& box : boxes) if (!separated(points, box)) return false;
        return true;
    };
    // Preserve every bit of a valid warm seed, including GPU arithmetic order.
    if (free(preferred)) return {preferred, preference, false};
    const Triangle fixed_edge{preferred[0], preferred[1], preferred[1]};
    for (const auto& box : boxes) if (!separated(fixed_edge, box)) {
        throw NoFeasibleSeed("PathCover fixed motion seed edge intersects an occupied voxel");
    }
    auto at = [&](double alpha) {
        return Triangle{preferred[0], preferred[1], tips[0] + alpha * direction};
    };
    if (length2 == 0.0 || domain.lower == domain.upper) {
        const auto point = at(domain.lower);
        if (free(point)) return {point, domain.lower, true};
        throw NoFeasibleSeed("PathCover motion tip segment has no collision-free seed");
    }
    std::vector<Interval> blocked;
    for (const auto& box : boxes) {
        const auto interval = collisionInterval(preferred[0], preferred[1], tips, box, domain);
        if (interval) blocked.push_back(*interval);
    }
    blocked = mergeIntervals(std::move(blocked));
    std::vector<Interval> available;
    double cursor = domain.lower;
    for (const auto& interval : blocked) {
        if (cursor < interval.lower) available.push_back({cursor, interval.lower});
        cursor = std::max(cursor, interval.upper);
    }
    if (cursor < domain.upper) available.push_back({cursor, domain.upper});
    std::sort(available.begin(), available.end(), [&](const Interval& a, const Interval& b) {
        const double da = std::abs(std::clamp(preference, a.lower, a.upper) - preference);
        const double db = std::abs(std::clamp(preference, b.lower, b.upper) - preference);
        return da < db || (da == db && a.lower < b.lower);
    });
    for (const auto& interval : available) {
        double rejected = std::clamp(preference, interval.lower, interval.upper);
        if (free(at(rejected))) return {at(rejected), rejected, true};
        double accepted = interval.lower + (interval.upper - interval.lower) * 0.5;
        if (!free(at(accepted))) {
            // A sub-tolerance gap is not a usable RISP seed. Try its physical
            // interval endpoints as well; contact endpoints fail the same SAT.
            if (free(at(interval.lower))) accepted = interval.lower;
            else if (free(at(interval.upper))) accepted = interval.upper;
            else continue;
        }
        // Geometric free intervals are open at obstacle contacts, so an exact
        // nearest free point need not exist. Project inward to a point passing
        // the existing SAT gap, within sqrt(machine epsilon) in alpha. This
        // only reads cached local boxes; no LP, map scan or replanning retry.
        const double tolerance = std::sqrt(std::numeric_limits<double>::epsilon());
        while (std::abs(accepted - rejected) > tolerance) {
            const double middle = rejected + (accepted - rejected) * 0.5;
            if (free(at(middle))) accepted = middle;
            else rejected = middle;
        }
        return {at(accepted), accepted, true};
    }
    if (available.empty()) {
        throw NoFeasibleSeed("PathCover motion tip segment has no collision-free seed");
    }
    throw NoFeasibleSeed("PathCover motion tip segment has no numerically separated seed");
}

inline Selection select(const Triangle& preferred, const TipSegment& tips,
                        const pathcover_endpoint::FineGeometry& geometry) {
    const Interval domain = workspaceDomain(preferred, tips, geometry.lower(), geometry.upper());
    const Point direction = tips[1] - tips[0];
    Point minimum = preferred[0].cwiseMin(preferred[1]);
    Point maximum = preferred[0].cwiseMax(preferred[1]);
    for (double alpha : {domain.lower, domain.upper}) {
        const Point point = tips[0] + alpha * direction;
        minimum = minimum.cwiseMin(point); maximum = maximum.cwiseMax(point);
    }
    const double resolution = geometry.resolution();
    const Eigen::Vector3i first = (minimum.array() / resolution).floor().cast<int>().matrix()
        - Eigen::Vector3i::Ones();
    const Eigen::Vector3i last = (maximum.array() / resolution).floor().cast<int>().matrix()
        + Eigen::Vector3i::Ones();
    std::vector<Box> boxes;
    geometry.visitOccupied(first, last, [&](const Eigen::Vector3i& index) {
        const Point lower = index.cast<double>() * resolution;
        const Point upper = (index + Eigen::Vector3i::Ones()).cast<double>() * resolution;
        if ((upper.array() >= minimum.array()).all() && (lower.array() <= maximum.array()).all()) {
            boxes.push_back({lower, upper});
        }
        return true;
    });
    return select(preferred, tips, boxes, geometry.lower(), geometry.upper());
}

} // namespace pathcover_motion_seed
