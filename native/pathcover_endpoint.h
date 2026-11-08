#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "occupancy_inflation/voxel_bit_grid.h"
#include "pathcover_clearance.h"

namespace pathcover_endpoint {

using Point = Eigen::Vector3d;
using Index = Eigen::Vector3i;

// Read the same logical inflated voxels used to construct the search grid.
// No packed-float serialization or second body inflation is involved.
class FineGeometry {
    const occupancy_inflation::VoxelBitGrid& grid_;
    const occupancy_inflation::VoxelBitGrid* source_grid_;
    double resolution_;
    Point lower_, upper_;
    Point body_radii_;

public:
    FineGeometry(const occupancy_inflation::VoxelBitGrid& grid, double resolution,
                 const Point& lower, const Point& upper,
                 const occupancy_inflation::VoxelBitGrid* source_grid = nullptr,
                 const Point& body_radii = Point::Zero())
        : grid_(grid), source_grid_(source_grid), resolution_(resolution),
          lower_(lower), upper_(upper), body_radii_(body_radii) {}

    double resolution() const { return resolution_; }
    const Point& lower() const { return lower_; }
    const Point& upper() const { return upper_; }
    const Point& bodyRadii() const { return body_radii_; }
    bool usesBoxEllipsoid() const { return source_grid_ != nullptr; }
    Point center(const Index& index) const {
        return (index.cast<double>().array() + 0.5).matrix() * resolution_;
    }
    bool insideWorkspace(const Point& point) const {
        return (point.array() >= lower_.array()).all()
            && (point.array() <= upper_.array()).all();
    }
    bool centerFree(const Index& index) const {
        if (source_grid_) return pointFree(center(index));
        return insideWorkspace(center(index)) && !grid_.testGlobal(index);
    }

    template<class Visitor>
    static bool visitGridOccupied(const occupancy_inflation::VoxelBitGrid& grid,
                                  Index first, Index last, Visitor&& visitor) {
        if (grid.empty()) return true;
        const auto& window = grid.window();
        first = first.cwiseMax(window.base);
        last = last.cwiseMin(window.maxIndex());
        if ((first.array() > last.array()).any()) return true;
        const Index local_first = first - window.base, local_last = last - window.base;
        const int first_word = local_first.x() / occupancy_inflation::kWordBits;
        const int last_word = local_last.x() / occupancy_inflation::kWordBits;
        for (int z = local_first.z(); z <= local_last.z(); ++z) {
            for (int y = local_first.y(); y <= local_last.y(); ++y) {
                const auto* row = grid.data() + window.rowOffset(y, z);
                for (int word = first_word; word <= last_word; ++word) {
                    uint64_t bits = row[word];
                    if (word == first_word) bits &= ~uint64_t{0} << (local_first.x() % 64);
                    const unsigned high = static_cast<unsigned>(local_last.x() % 64);
                    if (word == last_word && high != 63) bits &= (uint64_t{1} << (high + 1)) - 1;
                    while (bits) {
                        const int bit = __builtin_ctzll(bits);
                        bits &= bits - 1;
                        const Index index(window.base.x() + word * 64 + bit,
                                          window.base.y() + y, window.base.z() + z);
                        if (!visitor(index)) return false;
                    }
                }
            }
        }
        return true;
    }

    template<class Visitor>
    bool visitOccupied(Index first, Index last, Visitor&& visitor) const {
        return visitGridOccupied(grid_, first, last, std::forward<Visitor>(visitor));
    }

    template<class Visitor>
    bool visitSourceOccupied(Index first, Index last, Visitor&& visitor) const {
        return visitGridOccupied(source_grid_ ? *source_grid_ : grid_, first, last,
                                 std::forward<Visitor>(visitor));
    }

    std::vector<Point> occupiedCenters() const {
        std::vector<Point> centers;
        if (grid_.empty()) return centers;
        centers.reserve(grid_.count());
        visitOccupied(grid_.window().base, grid_.window().maxIndex(), [&](const Index& index) {
            centers.push_back(center(index));
            return true;
        });
        return centers;
    }

    static bool segmentIntersectsBox(const Point& begin, const Point& end,
                                     const Point& center, double half) {
        double first = 0.0, last = 1.0;
        const Point direction = end - begin;
        for (int axis = 0; axis < 3; ++axis) {
            const double lower = center[axis] - half, upper = center[axis] + half;
            if (direction[axis] == 0.0) {
                if (begin[axis] < lower || begin[axis] > upper) return false;
            } else {
                double a = (lower - begin[axis]) / direction[axis];
                double b = (upper - begin[axis]) / direction[axis];
                if (a > b) std::swap(a, b);
                first = std::max(first, a);
                last = std::min(last, b);
                if (first > last) return false;
            }
        }
        return true; // Closed AABBs: face, edge and corner contacts are blocked.
    }

    bool segmentFree(const Point& begin, const Point& end) const {
        if (!insideWorkspace(begin) || !insideWorkspace(end)) return false;
        // The one-index halo includes boxes touching a grid face. It only
        // broadens enumeration; it does not enlarge the occupied geometry.
        const Point segment_lower = begin.cwiseMin(end), segment_upper = begin.cwiseMax(end);
        const Index first = (segment_lower.array() / resolution_).floor().cast<int>().matrix()
            - Index::Ones();
        const Index last = (segment_upper.array() / resolution_).floor().cast<int>().matrix()
            + Index::Ones();
        if (!source_grid_) {
            return visitOccupied(first, last, [&](const Index& index) {
                return !segmentIntersectsBox(begin, end, center(index), resolution_ * 0.5);
            });
        }
        // New-mode raster bounds enclose the integer lattice endpoints rather
        // than reconstructing center +/- half, which can shrink at closed
        // contact. One outward ULP also covers products such as 7 * .1 > .7.
        // A segment-AABB overlap is deliberately only a conservative broad
        // phase; avoiding slab divisions here preserves that inclusion at a
        // tangency. Every possible hit is decided by the raw narrow phase.
        // The old per-voxel AABB predicate separates by axis. Trim each
        // integer interval once using the same outward-rounded lattice bounds,
        // then test its bitmap words without reconstructing occupied voxels.
        Index raster_first = first, raster_last = last;
        for (int axis = 0; axis < 3; ++axis) {
            while (raster_first[axis] <= raster_last[axis]
                   && pathcover_clearance::detail::up(
                       static_cast<double>(raster_first[axis] + 1) * resolution_)
                       < segment_lower[axis]) ++raster_first[axis];
            while (raster_last[axis] >= raster_first[axis]
                   && pathcover_clearance::detail::down(
                       static_cast<double>(raster_last[axis]) * resolution_)
                       > segment_upper[axis]) --raster_last[axis];
        }
        const bool raster_free = visitOccupied(raster_first, raster_last,
                                               [](const Index&) { return false; });
        if (raster_free) return true;
        // The raster is a conservative broad phase. Narrow-phase clearance is
        // measured against raw occupied boxes plus the body ellipsoid once.
        const Index raw_first = ((segment_lower - body_radii_).array()
            / resolution_).floor().cast<int>().matrix() - Index::Ones();
        const Index raw_last = ((segment_upper + body_radii_).array()
            / resolution_).floor().cast<int>().matrix() + Index::Ones();
        return visitSourceOccupied(raw_first, raw_last, [&](const Index& index) {
            const pathcover_clearance::Box box{
                index.cast<double>() * resolution_,
                (index + Index::Ones()).cast<double>() * resolution_};
            return pathcover_clearance::segmentFree(begin, end, box, body_radii_);
        });
    }

    bool pointFree(const Point& point) const { return segmentFree(point, point); }
};

struct CoarseGrid {
    const std::vector<signed char>& cells;
    Index dimensions;
    Point origin;
    double resolution;

    Index index(const Point& point) const {
        // Match JPS::MapUtil::floatToInt, including its boundary convention.
        Index result;
        for (int axis = 0; axis < 3; ++axis) {
            result[axis] = static_cast<int>(std::round((point[axis] - origin[axis]) / resolution - 0.5));
        }
        return result;
    }
    int linear(const Index& index) const {
        return index.x() + dimensions.x() * (index.y() + dimensions.y() * index.z());
    }
    bool free(const Index& index) const {
        return (index.array() >= 0).all() && (index.array() < dimensions.array()).all()
            && cells[linear(index)] < 100;
    }
    Point center(const Index& index) const {
        return origin + (index.cast<double>().array() + 0.5).matrix() * resolution;
    }
};

struct Connection {
    Index coarse_index;
    std::vector<Point> points; // Endpoint -> coarse-cell center.
    double length = 0.0;
};

struct Connections {
    std::vector<Connection> anchors;
    std::size_t visited_cells = 0;
    bool endpoint_in_coarse_free_cell = false;
};

inline std::vector<Point> simplify(const std::vector<Point>& path, const FineGeometry& fine,
                                   const std::vector<int>* fine_node_ids = nullptr,
                                   std::vector<int8_t>* endpoint_visibility = nullptr) {
    std::vector<Point> result;
    if (path.empty()) return result;
    result.push_back(path.front());
    // Only an actual fine-node tag is eligible. Endpoint/coarse vertices carry
    // -1; never infer this tag by rounding coordinates or reverse the segment.
    const auto visible = [&](std::size_t from, std::size_t to) {
        if (from == 0 && fine_node_ids && endpoint_visibility
            && (*fine_node_ids)[to] >= 0) {
            auto& known = (*endpoint_visibility)[(*fine_node_ids)[to]];
            if (known < 0)
                known = fine.segmentFree(path[from], path[to]) ? 1 : 0;
            return known != 0;
        }
        return fine.segmentFree(path[from], path[to]);
    };
    for (std::size_t i = 0; i + 1 < path.size();) {
        std::size_t next = path.size() - 1;
        while (!visible(i, next)) {
            if (next == i + 1) {
                throw std::logic_error("Endpoint connector contains an invalid fine-grid edge");
            }
            --next;
        }
        if ((path[next] - result.back()).squaredNorm() > 0.0) result.push_back(path[next]);
        i = next;
    }
    return result;
}

inline Connections connect(const Point& endpoint, const FineGeometry& fine,
                           const CoarseGrid& coarse, double radius) {
    if (!endpoint.allFinite() || !std::isfinite(radius) || radius <= 0.0) {
        throw std::invalid_argument("Endpoint connection requires a finite point and positive radius");
    }
    Connections result;
    if (!fine.pointFree(endpoint)) return result;
    const Index endpoint_coarse = coarse.index(endpoint);
    result.endpoint_in_coarse_free_cell = coarse.free(endpoint_coarse);
    if (result.endpoint_in_coarse_free_cell) {
        const Point center = coarse.center(endpoint_coarse);
        if (!fine.segmentFree(endpoint, center)) {
            throw std::logic_error("Free coarse cell failed its fine-geometry inclusion contract");
        }
        result.anchors.push_back({endpoint_coarse, {endpoint, center}, (center - endpoint).norm()});
        return result;
    }

    const double resolution = fine.resolution();
    const Point window_lower = (endpoint.array() - radius).matrix().cwiseMax(fine.lower());
    const Point window_upper = (endpoint.array() + radius).matrix().cwiseMin(fine.upper());
    const Index first = (window_lower.array() / resolution - 0.5).ceil().cast<int>().matrix();
    const Index last = (window_upper.array() / resolution - 0.5).floor().cast<int>().matrix();
    const Index dimensions = last - first + Index::Ones();
    if ((dimensions.array() <= 0).any()) return result;
    const int64_t count64 = int64_t(dimensions.x()) * dimensions.y() * dimensions.z();
    if (count64 > std::numeric_limits<int>::max()) {
        throw std::invalid_argument("Endpoint connection window exceeds integer indexing capacity");
    }
    const int count = static_cast<int>(count64);
    auto localIndex = [&](const Index& global) {
        const Index local = global - first;
        return local.x() + dimensions.x() * (local.y() + dimensions.y() * local.z());
    };
    auto globalIndex = [&](int id) -> Index {
        return first + Index(id % dimensions.x(), (id / dimensions.x()) % dimensions.y(),
                             id / (dimensions.x() * dimensions.y()));
    };
    auto inWindow = [&](const Index& index) {
        return (index.array() >= first.array()).all() && (index.array() <= last.array()).all();
    };
    // Six-axis edges are equal length, but virtual endpoint edges are not.
    // A local Dijkstra search preserves the shortest grid connection to each
    // anchor. It exhausts this fixed window; no radius growth or retry occurs.
    // -2 unseen, -3 blocked, -1 a root connected directly to the real endpoint.
    std::vector<int> parents(count, -2);
    std::vector<double> distance(count, std::numeric_limits<double>::infinity());
    using Pending = std::pair<double, int>;
    std::priority_queue<Pending, std::vector<Pending>, std::greater<Pending>> queue;
    const Index containing = (endpoint.array() / resolution).floor().cast<int>().matrix();
    // A physical-boundary point may have its containing cell center outside
    // the usable workspace. Visible adjacent centers represent that endpoint
    // without moving it or enlarging the physical bounds.
    for (int z = -1; z <= 1; ++z) for (int y = -1; y <= 1; ++y) for (int x = -1; x <= 1; ++x) {
        const Index index = containing + Index(x, y, z);
        if (!inWindow(index)) continue;
        const int id = localIndex(index);
        if (fine.segmentFree(endpoint, fine.center(index))) {
            parents[id] = -1;
            distance[id] = (fine.center(index) - endpoint).norm();
            queue.emplace(distance[id], id);
        }
    }
    const std::array<Index, 6> directions{
        Index(-1,0,0), Index(1,0,0), Index(0,-1,0), Index(0,1,0), Index(0,0,-1), Index(0,0,1)};
    struct AnchorNode { Index coarse_index; int node; double length; };
    std::vector<AnchorNode> anchor_nodes;
    std::unordered_map<int, std::size_t> anchor_indices;
    while (!queue.empty()) {
        const auto [cost, id] = queue.top();
        queue.pop();
        if (cost > distance[id]) continue;
        const Index global = globalIndex(id);
        const Point point = fine.center(global);
        ++result.visited_cells;
        const Index coarse_index = coarse.index(point);
        if (coarse.free(coarse_index)) {
            const Point target = coarse.center(coarse_index);
            const bool target_in_window = (target.array() >= window_lower.array()).all()
                && (target.array() <= window_upper.array()).all();
            if (target_in_window) {
                const double length = cost + (target - point).norm();
                const int anchor_id = coarse.linear(coarse_index);
                const auto found = anchor_indices.find(anchor_id);
                // Only a strictly shorter candidate can replace a valid anchor.
                if ((found == anchor_indices.end() || length < anchor_nodes[found->second].length)
                    && fine.segmentFree(point, target)) {
                    if (found == anchor_indices.end()) {
                        anchor_indices.emplace(anchor_id, anchor_nodes.size());
                        anchor_nodes.push_back({coarse_index, id, length});
                    } else {
                        anchor_nodes[found->second] = {coarse_index, id, length};
                    }
                }
            }
        }
        for (const Index& direction : directions) {
            const Index next = global + direction;
            if (!inWindow(next)) continue;
            const int next_id = localIndex(next);
            if (parents[next_id] == -3) continue;
            if (parents[next_id] == -2 && !fine.centerFree(next)) {
                parents[next_id] = -3;
                continue;
            }
            const double candidate = cost + (fine.center(next) - point).norm();
            // Raw boxes use this same lattice and the ellipsoid is axis aligned.
            // Any nonempty cross-section along this axis contains the raw box's
            // interval (at least one cell wide). A one-cell center-to-center edge
            // cannot cross that section with both endpoints free. This proof
            // does not apply to endpoint, anchor, or simplified long segments.
            if (candidate < distance[next_id]) {
                parents[next_id] = id;
                distance[next_id] = candidate;
                queue.emplace(candidate, next_id);
            }
        }
    }
    // Reconstruct only one shortest local-grid route per coarse anchor, then
    // remove unnecessary turns using the selected exact geometry segment test.
    // Each connect owns this result table; no map/endpoint state survives.
    std::vector<int8_t> endpoint_visibility(anchor_nodes.empty() ? 0 : count, -1);
    for (const auto& anchor : anchor_nodes) {
        std::vector<Point> reverse;
        std::vector<int> reverse_nodes;
        for (int node = anchor.node; node >= 0; node = parents[node]) {
            reverse.push_back(fine.center(globalIndex(node)));
            reverse_nodes.push_back(node);
        }
        std::vector<Point> path{endpoint};
        std::vector<int> path_nodes{-1};
        path.insert(path.end(), reverse.rbegin(), reverse.rend());
        path_nodes.insert(path_nodes.end(), reverse_nodes.rbegin(), reverse_nodes.rend());
        path.push_back(coarse.center(anchor.coarse_index));
        path_nodes.push_back(-1);
        path = simplify(path, fine, &path_nodes, &endpoint_visibility);
        double length = 0.0;
        for (std::size_t i = 1; i < path.size(); ++i) length += (path[i] - path[i-1]).norm();
        result.anchors.push_back({anchor.coarse_index, std::move(path), length});
    }
    std::sort(result.anchors.begin(), result.anchors.end(), [&](const Connection& a, const Connection& b) {
        return coarse.linear(a.coarse_index) < coarse.linear(b.coarse_index);
    });
    return result;
}

} // namespace pathcover_endpoint
