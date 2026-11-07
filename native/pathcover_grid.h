#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <limits>
#include <vector>

namespace pathcover_grid {

// Preserve the original fine-voxel AABB overlap, including its floating-point
// boundary convention. The caller resets previous occupancy before this call
// and sorts occupied_indices afterwards. Virtual boundary cells stay occupied.
inline void coarsenOccupiedVolumes(
    const std::vector<Eigen::Vector3d>& points,
    const Eigen::Vector3d& origin,
    double fine_resolution,
    double search_resolution,
    const Eigen::Vector3i& dimensions,
    std::vector<signed char>& cells,
    std::vector<int>& occupied_indices) {
    const double half = 0.5 * fine_resolution;
    // For max(0, floor(v)), negative values all clamp to zero. For ceil(v),
    // truncation needs one increment exactly when v exceeds the integer value.
    auto lowerIndex = [](double value) {
        return std::max(0, static_cast<int>(value));
    };
    auto upperIndex = [](double value, int maximum) {
        const int truncated = static_cast<int>(value);
        return std::min(maximum, truncated + static_cast<int>(value > truncated) - 1);
    };
    auto occupy = [&](const Eigen::Vector3i& lo, const Eigen::Vector3i& hi) {
        for (int z = lo.z(); z <= hi.z(); ++z) {
            for (int y = lo.y(); y <= hi.y(); ++y) {
                const int row = dimensions.x() * (y + dimensions.y() * z);
                for (int x = lo.x(); x <= hi.x(); ++x) {
                    const int index = row + x;
                    if (cells[index] == 0) {
                        cells[index] = 100;
                        occupied_indices.push_back(index);
                    }
                }
            }
        }
    };
    Eigen::Vector3i run_lo, run_hi, cached_lo, cached_hi;
    bool have_run = false;
    double previous_y = std::numeric_limits<double>::quiet_NaN();
    double previous_z = std::numeric_limits<double>::quiet_NaN();
    for (const auto& point : points) {
        Eigen::Vector3i lo, hi;
        const double local_x = point.x() - origin.x();
        lo.x() = lowerIndex((local_x - half) / search_resolution + 1e-9);
        hi.x() = upperIndex((local_x + half) / search_resolution - 1e-9, dimensions.x() - 1);
        if (point.y() != previous_y) {
            const double local_y = point.y() - origin.y();
            cached_lo.y() = lowerIndex((local_y - half) / search_resolution + 1e-9);
            cached_hi.y() = upperIndex((local_y + half) / search_resolution - 1e-9, dimensions.y() - 1);
            previous_y = point.y();
        }
        if (point.z() != previous_z) {
            const double local_z = point.z() - origin.z();
            cached_lo.z() = lowerIndex((local_z - half) / search_resolution + 1e-9);
            cached_hi.z() = upperIndex((local_z + half) / search_resolution - 1e-9, dimensions.z() - 1);
            previous_z = point.z();
        }
        lo.y() = cached_lo.y(); lo.z() = cached_lo.z();
        hi.y() = cached_hi.y(); hi.z() = cached_hi.z();
        if ((lo.array() > hi.array()).any()) continue;
        // Merge only overlapping/adjacent *coarse index* intervals with identical
        // Y/Z coverage. This is an exact union, even for unordered input points.
        if (have_run && lo.y() == run_lo.y() && lo.z() == run_lo.z()
            && hi.y() == run_hi.y() && hi.z() == run_hi.z()
            && lo.x() <= run_hi.x() + 1 && hi.x() >= run_lo.x() - 1) {
            run_lo.x() = std::min(run_lo.x(), lo.x());
            run_hi.x() = std::max(run_hi.x(), hi.x());
        } else {
            if (have_run) occupy(run_lo, run_hi);
            run_lo = lo; run_hi = hi; have_run = true;
        }
    }
    if (have_run) occupy(run_lo, run_hi);
}

} // namespace pathcover_grid
