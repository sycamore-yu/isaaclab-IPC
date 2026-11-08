#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

namespace pathcover_clearance {

using Vec3 = Eigen::Vector3d;
using Triangle = std::array<Vec3, 3>;
using Tetrahedron = std::array<Vec3, 4>;

struct Box { Vec3 lower, upper; };
struct Interval { double lower, upper; };
struct Closest {
    double squared_distance = std::numeric_limits<double>::infinity();
    Vec3 simplex_point = Vec3::Zero();
    Vec3 box_point = Vec3::Zero();
};
struct Clearance {
    // Squared Euclidean distance after diag(1/r) for positive radii, or the
    // unscaled distance for exactly zero radii. Witnesses remain world points.
    double squared_distance = 0.0;
    Vec3 simplex_point = Vec3::Zero(), box_point = Vec3::Zero();
    // Unit world normal from obstacle toward simplex. For free results its
    // support bounds give a separating plane without an extra body inflation.
    Vec3 normal = Vec3::Zero();
    double seed_support = 0.0, obstacle_support = 0.0;
    bool free = false;
};

namespace detail {
using Wide3 = Eigen::Matrix<long double, 3, 1>;

inline void validate(const Box& box) {
    if (!box.lower.allFinite() || !box.upper.allFinite()
        || (box.lower.array() > box.upper.array()).any())
        throw std::invalid_argument("Clearance requires a finite ordered AABB");
}
template<std::size_t N>
inline void validate(const std::array<Vec3, N>& points, const Box& box) {
    validate(box);
    for (const auto& p : points)
        if (!p.allFinite()) throw std::invalid_argument("Clearance requires finite points");
}
inline Closest pair(const Vec3& p, const Vec3& q) {
    const long double d2 = (p.cast<long double>() - q.cast<long double>()).squaredNorm();
    if (!p.allFinite() || !q.allFinite() || !std::isfinite(d2)
        || d2 > std::numeric_limits<double>::max())
        throw std::overflow_error("Clearance distance is outside the finite numeric range");
    return {static_cast<double>(d2), p, q};
}
inline void take(Closest& best, const Closest& value) {
    if (value.squared_distance < best.squared_distance) best = value;
}
inline Vec3 clampBox(const Vec3& p, const Box& box) {
    return p.cwiseMax(box.lower).cwiseMin(box.upper);
}
inline Vec3 segmentPoint(const Vec3& p, const Vec3& a, const Vec3& b) {
    const Wide3 u = b.cast<long double>() - a.cast<long double>();
    const long double length2 = u.squaredNorm();
    if (length2 == 0.0L) return a;
    const long double t = std::clamp(
        (p.cast<long double>() - a.cast<long double>()).dot(u) / length2, 0.0L, 1.0L);
    return (a.cast<long double>() + t * u).cast<double>();
}
inline Closest segmentSegment(const Vec3& a, const Vec3& b,
                              const Vec3& c, const Vec3& d) {
    Closest best = pair(a, segmentPoint(a, c, d));
    take(best, pair(b, segmentPoint(b, c, d)));
    take(best, pair(segmentPoint(c, a, b), c));
    take(best, pair(segmentPoint(d, a, b), d));
    const Wide3 u = b.cast<long double>() - a.cast<long double>();
    const Wide3 v = d.cast<long double>() - c.cast<long double>();
    const Wide3 r = a.cast<long double>() - c.cast<long double>();
    const Wide3 cross = u.cross(v);
    const long double denominator = cross.squaredNorm();
    // The cross-product denominator avoids cancellation in |u|²|v|²-(u.v)².
    // Parallel and zero-length cases already have an endpoint minimizer above.
    if (denominator > 0.0L) {
        const long double s = v.cross(r).dot(cross) / denominator;
        const long double t = u.cross(r).dot(cross) / denominator;
        if (s >= 0.0L && s <= 1.0L && t >= 0.0L && t <= 1.0L)
            take(best, pair((a.cast<long double>() + s * u).cast<double>(),
                            (c.cast<long double>() + t * v).cast<double>()));
    }
    return best;
}
inline Vec3 trianglePoint(const Vec3& p, const Triangle& triangle) {
    Vec3 best = segmentPoint(p, triangle[0], triangle[1]);
    double best_d2 = pair(p, best).squared_distance;
    for (int i = 1; i < 3; ++i) {
        const Vec3 candidate = segmentPoint(p, triangle[i], triangle[(i + 1) % 3]);
        const double d2 = pair(p, candidate).squared_distance;
        if (d2 < best_d2) { best = candidate; best_d2 = d2; }
    }
    const Wide3 a = triangle[0].cast<long double>();
    const Wide3 b = triangle[1].cast<long double>();
    const Wide3 c = triangle[2].cast<long double>();
    const Wide3 normal = (b - a).cross(c - a);
    const long double norm2 = normal.squaredNorm();
    // A truly degenerate triangle is the union of its edges. Do not collapse a
    // thin, nonzero-area triangle using a fixed world-scale epsilon.
    if (norm2 > 0.0L) {
        const Wide3 query = p.cast<long double>();
        const Wide3 projected = query - normal * ((query - a).dot(normal) / norm2);
        if ((b-a).cross(projected-a).dot(normal) >= 0.0L
            && (c-b).cross(projected-b).dot(normal) >= 0.0L
            && (a-c).cross(projected-c).dot(normal) >= 0.0L) {
            const Vec3 candidate = projected.cast<double>();
            if (pair(p, candidate).squared_distance < best_d2) best = candidate;
        }
    }
    return best;
}
inline std::array<Vec3, 8> vertices(const Box& box) {
    std::array<Vec3, 8> result;
    for (int i = 0; i < 8; ++i)
        for (int axis = 0; axis < 3; ++axis)
            result[i][axis] = i & (1 << axis) ? box.upper[axis] : box.lower[axis];
    return result;
}
inline bool segmentBoxIntersection(const Vec3& a, const Vec3& b,
                                   const Box& box, Vec3& witness) {
    long double first = 0.0L, last = 1.0L;
    const Wide3 start = a.cast<long double>();
    const Wide3 direction = b.cast<long double>() - start;
    for (int axis = 0; axis < 3; ++axis) {
        if (direction[axis] == 0.0L) {
            if (start[axis] < box.lower[axis] || start[axis] > box.upper[axis]) return false;
        } else {
            long double lo = (static_cast<long double>(box.lower[axis]) - start[axis]) / direction[axis];
            long double hi = (static_cast<long double>(box.upper[axis]) - start[axis]) / direction[axis];
            if (lo > hi) std::swap(lo, hi);
            first = std::max(first, lo);
            last = std::min(last, hi);
            if (first > last) return false;
        }
    }
    witness = (start + first * direction).cast<double>();
    return true;
}
inline bool triangleBoxSAT(const Triangle& triangle, const Box& box) {
    const Wide3 center = (box.lower.cast<long double>() + box.upper.cast<long double>()) * 0.5L;
    const Wide3 half = (box.upper.cast<long double>() - box.lower.cast<long double>()) * 0.5L;
    std::array<Wide3, 3> p;
    for (int i = 0; i < 3; ++i) p[i] = triangle[i].cast<long double>() - center;
    const auto separates = [&](const Wide3& axis) {
        if (axis.squaredNorm() == 0.0L) return false;
        const long double a = axis.dot(p[0]), b = axis.dot(p[1]), c = axis.dot(p[2]);
        const long double radius = axis.cwiseAbs().dot(half);
        return std::min({a,b,c}) > radius || std::max({a,b,c}) < -radius;
    };
    for (int axis = 0; axis < 3; ++axis)
        if (separates(Wide3::Unit(axis))) return false;
    if (separates((p[1]-p[0]).cross(p[2]-p[0]))) return false;
    for (int edge = 0; edge < 3; ++edge)
        for (int axis = 0; axis < 3; ++axis)
            if (separates((p[(edge+1)%3]-p[edge]).cross(Wide3::Unit(axis)))) return false;
    return true;
}
inline Vec3 triangleBoxWitness(const Triangle& triangle, const Box& box) {
    // Clip only after SAT reports intersection. This also handles point/line
    // degeneracy and produces an actual common point instead of a fake center.
    std::vector<Wide3> polygon;
    for (const auto& p : triangle) polygon.push_back(p.cast<long double>());
    for (int axis = 0; axis < 3; ++axis) {
        for (int side = 0; side < 2; ++side) {
            const long double bound = side ? box.upper[axis] : box.lower[axis];
            const auto signed_distance = [&](const Wide3& p) {
                return side ? p[axis] - bound : bound - p[axis];
            };
            std::vector<Wide3> clipped;
            if (polygon.empty()) break;
            Wide3 previous = polygon.back();
            long double dp = signed_distance(previous);
            for (const auto& current : polygon) {
                const long double dc = signed_distance(current);
                if ((dp <= 0.0L) != (dc <= 0.0L)) {
                    Wide3 intersection = previous + (current - previous) * (dp / (dp - dc));
                    intersection[axis] = bound;
                    clipped.push_back(intersection);
                }
                if (dc <= 0.0L) clipped.push_back(current);
                previous = current;
                dp = dc;
            }
            polygon = std::move(clipped);
        }
    }
    if (polygon.empty())
        throw std::runtime_error("Triangle/AABB predicates disagree at numeric precision");
    return polygon.front().cast<double>();
}
inline Closest pointBox(const Vec3& p, const Box& box) {
    return pair(p, clampBox(p, box));
}
inline Closest segmentBox(const Vec3& a, const Vec3& b, const Box& box) {
    Vec3 witness;
    if (segmentBoxIntersection(a, b, box, witness)) return {0.0, witness, witness};
    Closest best = pointBox(a, box);
    take(best, pointBox(b, box));
    const auto corners = vertices(box);
    for (int i = 0; i < 8; ++i)
        for (int axis = 0; axis < 3; ++axis)
            if (!(i & (1 << axis))) take(best, segmentSegment(a, b, corners[i], corners[i | (1 << axis)]));
    return best;
}
inline Closest triangleBox(const Triangle& triangle, const Box& box) {
    if (triangleBoxSAT(triangle, box)) {
        const Vec3 witness = triangleBoxWitness(triangle, box);
        return {0.0, witness, witness};
    }
    Closest best;
    for (const auto& p : triangle) take(best, pointBox(p, box));
    const auto corners = vertices(box);
    for (const auto& q : corners) take(best, pair(trianglePoint(q, triangle), q));
    for (int edge = 0; edge < 3; ++edge)
        for (int i = 0; i < 8; ++i)
            for (int axis = 0; axis < 3; ++axis)
                if (!(i & (1 << axis))) take(best, segmentSegment(
                    triangle[edge], triangle[(edge+1)%3], corners[i], corners[i | (1 << axis)]));
    return best;
}
inline bool contains(const Tetrahedron& tetra, const Vec3& point) {
    // Nonzero signed volume is required. A coplanar/collinear tetrahedron is
    // covered by its four triangular faces, without division by its volume.
    const Wide3 p = point.cast<long double>();
    for (int opposite = 0; opposite < 4; ++opposite) {
        std::array<int, 3> indices;
        int count = 0;
        for (int i = 0; i < 4; ++i) if (i != opposite) indices[count++] = i;
        const Wide3 a = tetra[indices[0]].cast<long double>();
        const Wide3 normal = (tetra[indices[1]].cast<long double>() - a)
            .cross(tetra[indices[2]].cast<long double>() - a);
        const long double direction = normal.dot(tetra[opposite].cast<long double>() - a);
        const long double query = normal.dot(p - a);
        if (direction == 0.0L || (direction > 0.0L ? query < 0.0L : query > 0.0L)) return false;
    }
    return true;
}
inline Closest tetraBox(const Tetrahedron& tetra, const Box& box) {
    const Vec3 center = ((box.lower.cast<long double>() + box.upper.cast<long double>()) * 0.5L).cast<double>();
    if (contains(tetra, center)) return {0.0, center, center};
    Closest best;
    for (int opposite = 0; opposite < 4; ++opposite) {
        Triangle face;
        int count = 0;
        for (int i = 0; i < 4; ++i) if (i != opposite) face[count++] = tetra[i];
        take(best, triangleBox(face, box));
        if (best.squared_distance == 0.0) break;
    }
    return best;
}
template<std::size_t N>
inline Closest closest(const std::array<Vec3, N>& points, const Box& box) {
    if constexpr (N == 1) return pointBox(points[0], box);
    else if constexpr (N == 2) return segmentBox(points[0], points[1], box);
    else if constexpr (N == 3) return triangleBox(points, box);
    else { static_assert(N == 4); return tetraBox(points, box); }
}

inline double down(double value) { return std::nextafter(value, -std::numeric_limits<double>::infinity()); }
inline double up(double value) { return std::nextafter(value, std::numeric_limits<double>::infinity()); }
inline double dotLower(const Vec3& a, const Vec3& b) {
    double value = 0.0;
    for (int i = 0; i < 3; ++i) value = down(value + down(a[i] * b[i]));
    return value;
}
inline double dotUpper(const Vec3& a, const Vec3& b) {
    double value = 0.0;
    for (int i = 0; i < 3; ++i) value = up(value + up(a[i] * b[i]));
    return value;
}
inline double ellipsoidSupportUpper(const Vec3& normal, const Vec3& radii) {
    if ((radii.array() == 0.0).all()) return 0.0;
    double sum = 0.0;
    for (int i = 0; i < 3; ++i) {
        const double component = up(std::abs(normal[i]) * radii[i]);
        sum = up(sum + up(component * component));
    }
    return up(std::sqrt(sum));
}
template<std::size_t N>
inline Clearance clearance(const std::array<Vec3, N>& points, const Box& box, const Vec3& radii) {
    validate(points, box);
    const bool zero = (radii.array() == 0.0).all();
    if (!radii.allFinite() || (!zero && !(radii.array() > 0.0).all()))
        throw std::invalid_argument("Body radii must be all positive or all zero");
    const Vec3 scale = zero ? Vec3::Ones() : radii;
    const Vec3 origin = ((box.lower.cast<long double>() + box.upper.cast<long double>()) * 0.5L).cast<double>();
    const Box transformed{(box.lower-origin).cwiseQuotient(scale), (box.upper-origin).cwiseQuotient(scale)};
    std::array<Vec3, N> transformed_points;
    for (std::size_t i = 0; i < N; ++i) transformed_points[i] = (points[i]-origin).cwiseQuotient(scale);
    validate(transformed_points, transformed);
    const Closest pair = closest(transformed_points, transformed);
    Clearance result;
    result.squared_distance = pair.squared_distance;
    result.simplex_point = origin + pair.simplex_point.cwiseProduct(scale);
    result.box_point = origin + pair.box_point.cwiseProduct(scale);
    if (pair.squared_distance == 0.0) return result;
    const Vec3 direction = (pair.simplex_point - pair.box_point).cwiseQuotient(scale);
    const double norm = direction.stableNorm();
    if (!(norm > 0.0) || !std::isfinite(norm))
        throw std::overflow_error("Clearance normal is outside the finite numeric range");
    // This support certificate is valid for any chosen normal. Roundoff in the
    // closest pair cannot turn contact into free space. Directed finite-term
    // bounds add no configurable radius; numerically ambiguous contact blocks.
    const auto certify = [&](const Vec3& candidate) {
        const double length = candidate.stableNorm();
        if (!(length > 0.0) || !std::isfinite(length)) return false;
        result.normal = candidate / length;
        result.seed_support = std::numeric_limits<double>::infinity();
        for (const auto& p : points) result.seed_support = std::min(result.seed_support, dotLower(result.normal, p));
        Vec3 corner;
        for (int i = 0; i < 3; ++i) corner[i] = result.normal[i] >= 0.0 ? box.upper[i] : box.lower[i];
        result.obstacle_support = dotUpper(result.normal, corner);
        if (!zero) result.obstacle_support = up(result.obstacle_support + ellipsoidSupportUpper(result.normal, radii));
        result.free = std::isfinite(result.seed_support) && std::isfinite(result.obstacle_support)
            && result.seed_support > result.obstacle_support;
        return result.free;
    };
    if (certify(direction)) return result;
    if (zero) {
        // At zero radius, nearly touching witnesses can lose an O(distance²)
        // coordinate when projected onto an edge. Use the original finite SAT
        // directions to certify a separating plane, rather than manufacturing
        // a small positive radius or claiming that rounded witnesses intersect.
        const auto either = [&](const Vec3& axis) { return certify(axis) || certify(-axis); };
        for (int axis = 0; axis < 3; ++axis)
            if (either(Vec3::Unit(axis))) return result;
        for (std::size_t i = 0; i < N; ++i) {
            for (std::size_t j = i+1; j < N; ++j) {
                const Wide3 edge = points[j].template cast<long double>() - points[i].template cast<long double>();
                for (int axis = 0; axis < 3; ++axis)
                    if (either(edge.cross(Wide3::Unit(axis)).template cast<double>())) return result;
                for (std::size_t k = j+1; k < N; ++k) {
                    const Wide3 other = points[k].template cast<long double>() - points[i].template cast<long double>();
                    if (either(edge.cross(other).template cast<double>())) return result;
                }
            }
        }
    }
    return result;
}
inline Vec3 tip(const std::array<Vec3, 2>& tips, double alpha) {
    if (alpha == 0.0) return tips[0];
    if (alpha == 1.0) return tips[1];
    return tips[0] + alpha * (tips[1] - tips[0]);
}
} // namespace detail

inline Closest pointBoxClosest(const Vec3& p, const Box& box) {
    detail::validate(std::array<Vec3, 1>{p}, box); return detail::pointBox(p, box);
}
inline Closest segmentBoxClosest(const Vec3& a, const Vec3& b, const Box& box) {
    detail::validate(std::array<Vec3, 2>{a,b}, box); return detail::segmentBox(a, b, box);
}
inline Closest triangleBoxClosest(const Triangle& triangle, const Box& box) {
    detail::validate(triangle, box); return detail::triangleBox(triangle, box);
}
inline Closest tetraBoxClosest(const Tetrahedron& tetra, const Box& box) {
    detail::validate(tetra, box); return detail::tetraBox(tetra, box);
}
inline Clearance triangleClosest(const Triangle& triangle, const Box& box, const Vec3& radii) {
    return detail::clearance(triangle, box, radii);
}
inline double boxSupportUpper(const Vec3& normal, const Box& box, const Vec3& radii) {
    detail::validate(box);
    const bool zero = (radii.array() == 0.0).all();
    if (!normal.allFinite() || !radii.allFinite()
        || (!zero && !(radii.array() > 0.0).all()))
        throw std::invalid_argument("Box support requires a finite normal and all-positive or zero radii");
    Vec3 corner;
    for (int i = 0; i < 3; ++i) corner[i] = normal[i] >= 0.0 ? box.upper[i] : box.lower[i];
    const double support = detail::dotUpper(normal, corner);
    return zero ? support : detail::up(support + detail::ellipsoidSupportUpper(normal, radii));
}
inline bool pointFree(const Vec3& p, const Box& box, const Vec3& radii) {
    return detail::clearance(std::array<Vec3, 1>{p}, box, radii).free;
}
inline bool segmentFree(const Vec3& a, const Vec3& b, const Box& box, const Vec3& radii) {
    return detail::clearance(std::array<Vec3, 2>{a,b}, box, radii).free;
}
inline bool triangleFree(const Triangle& triangle, const Box& box, const Vec3& radii) {
    return triangleClosest(triangle, box, radii).free;
}
inline bool tetraFree(const Tetrahedron& tetra, const Box& box, const Vec3& radii) {
    return detail::clearance(tetra, box, radii).free;
}

// Single convex obstacle only. With a fixed base, the collision parameters of
// conv(p0,p1,tip(alpha)) form an interval. Its swept set on [a,b] is exactly
// conv(p0,p1,tip(a),tip(b)); prefix/suffix tetra tests bracket both ends without
// discrete alpha samples. Multiple boxes must be merged by the caller.
inline std::optional<Interval> collisionInterval(
    const Vec3& p0, const Vec3& p1, const std::array<Vec3, 2>& tips,
    const Box& box, const Vec3& radii, Interval domain = {0.0, 1.0},
    double parameter_tolerance = 1e-10) {
    if (!std::isfinite(domain.lower) || !std::isfinite(domain.upper)
        || domain.lower < 0.0 || domain.upper > 1.0 || domain.lower > domain.upper
        || !std::isfinite(parameter_tolerance) || parameter_tolerance <= 0.0)
        throw std::invalid_argument("Collision interval requires an ordered [0,1] domain and positive tolerance");
    const Vec3 first = detail::tip(tips, domain.lower), last = detail::tip(tips, domain.upper);
    if (tetraFree(Tetrahedron{p0,p1,first,last}, box, radii)) return std::nullopt;
    if (!segmentFree(p0, p1, box, radii) || domain.lower == domain.upper) return domain;
    Interval blocked = domain;
    if (triangleFree(Triangle{p0,p1,first}, box, radii)) {
        double free_end = domain.lower, possible_end = domain.upper;
        for (int i = 0; i < 64 && possible_end - free_end > parameter_tolerance; ++i) {
            const double mid = free_end + 0.5 * (possible_end - free_end);
            if (mid == free_end || mid == possible_end) break;
            if (tetraFree(Tetrahedron{p0,p1,first,detail::tip(tips,mid)}, box, radii)) free_end = mid;
            else possible_end = mid;
        }
        blocked.lower = free_end; // Outer endpoint, never rounded into collision.
    }
    if (triangleFree(Triangle{p0,p1,last}, box, radii)) {
        double possible_begin = domain.lower, free_begin = domain.upper;
        for (int i = 0; i < 64 && free_begin - possible_begin > parameter_tolerance; ++i) {
            const double mid = possible_begin + 0.5 * (free_begin - possible_begin);
            if (mid == possible_begin || mid == free_begin) break;
            if (tetraFree(Tetrahedron{p0,p1,detail::tip(tips,mid),last}, box, radii)) free_begin = mid;
            else possible_begin = mid;
        }
        blocked.upper = free_begin;
    }
    if (blocked.lower > blocked.upper) return std::nullopt;
    return blocked;
}

} // namespace pathcover_clearance
