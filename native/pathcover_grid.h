#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "occupancy_inflation/voxel_bit_grid.h"
#include "pathcover_boxes.h"

namespace pathcover_grid {

// Recover a commensurate fine/search resolution ratio once at configuration.
// Division of decimal settings such as 0.1/0.3 can differ from 1/3 by one ULP.
// This recognizes the ratio within the arithmetic error, not a spatial epsilon.
// Unsupported ratios fail instead of silently quantizing arbitrary real scales.
using ResolutionRatio = std::pair<int64_t, int64_t>;
inline ResolutionRatio resolutionRatio(double fine, double search) {
    const double target = fine / search;
    const double ulp = std::nextafter(target, std::numeric_limits<double>::infinity()) - target;
    constexpr int64_t limit = 1000000;
    int64_t previous_numerator = 0, numerator = 1;
    int64_t previous_denominator = 1, denominator = 0;
    double remainder = target;
    for (int step = 0; step < 32; ++step) {
        const double integral = std::floor(remainder);
        if (!std::isfinite(integral) || integral > limit) break;
        const int64_t a = static_cast<int64_t>(integral);
        const int64_t next_numerator = a * numerator + previous_numerator;
        const int64_t next_denominator = a * denominator + previous_denominator;
        if (next_denominator > limit || next_numerator > limit) break;
        if (next_numerator > 0 && next_denominator > 0
            && std::abs(static_cast<double>(next_numerator) / next_denominator - target) <= 4.0 * ulp) {
            return {next_numerator, next_denominator};
        }
        previous_numerator = numerator; numerator = next_numerator;
        previous_denominator = denominator; denominator = next_denominator;
        const double fraction = remainder - integral;
        if (fraction == 0.0) break;
        remainder = 1.0 / fraction;
    }
    throw std::invalid_argument("PathCover voxel resolutions require a commensurate ratio with denominator <= 1000000");
}

// Project logical fine-cell volumes into a world-zero-anchored search grid.
// A coarse cell is occupied only for positive-volume intersection. In particular,
// a shared face/edge/corner does not occupy its neighbour. Read integer indices
// directly: packed float centres can move a touching face across a boundary.
// The caller resets cloud occupancy before this call and sorts indices afterwards.
// Already occupied virtual workspace boundary cells are preserved and not listed.
inline void coarsenOccupiedVolumes(
    const occupancy_inflation::VoxelBitGrid& fine_grid,
    const ResolutionRatio& resolution_ratio,
    const Eigen::Vector3i& search_origin_index,
    const Eigen::Vector3i& dimensions,
    std::vector<signed char>& cells,
    std::vector<int>& occupied_indices) {
    if (fine_grid.empty()) return;
    const auto& window = fine_grid.window();
    auto floorDivide = [](int64_t value, int64_t divisor) {
        return value / divisor - static_cast<int64_t>(value < 0 && value % divisor != 0);
    };
    struct Range { int lo; int hi; };
    std::array<std::vector<Range>, 3> ranges;
    for (int axis = 0; axis < 3; ++axis) {
        // Integer floor(lower) and ceil(upper)-1 implement strict interval
        // overlap, including negative indices and exactly shared boundaries.
        ranges[axis].resize(window.dims[axis]);
        for (int i = 0; i < window.dims[axis]; ++i) {
            const int64_t global = static_cast<int64_t>(window.base[axis]) + i;
            const int64_t lower = global * resolution_ratio.first;
            const int64_t upper = (global + 1) * resolution_ratio.first;
            const int64_t lo = floorDivide(lower, resolution_ratio.second) - search_origin_index[axis];
            const int64_t hi = -floorDivide(-upper, resolution_ratio.second) - 1 - search_origin_index[axis];
            ranges[axis][i] = {
                static_cast<int>(std::clamp<int64_t>(lo, 0, dimensions[axis])),
                static_cast<int>(std::clamp<int64_t>(hi, -1, dimensions[axis] - 1))};
        }
    }
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
    Eigen::Vector3i run_lo, run_hi;
    bool have_run = false;
    for (int z = 0; z < window.dims.z(); ++z) {
        const auto rz = ranges[2][z];
        if (rz.lo > rz.hi) continue;
        for (int y = 0; y < window.dims.y(); ++y) {
            const auto ry = ranges[1][y];
            if (ry.lo > ry.hi) continue;
            const auto* row = fine_grid.data() + window.rowOffset(y, z);
            for (std::size_t word = 0; word < window.wordsPerRow(); ++word) {
                uint64_t bits = row[word];
                while (bits) {
                    const int bit = __builtin_ctzll(bits);
                    bits &= bits - 1;
                    const auto rx = ranges[0][word * occupancy_inflation::kWordBits + bit];
                    if (rx.lo > rx.hi) continue;
                    const Eigen::Vector3i lo(rx.lo, ry.lo, rz.lo), hi(rx.hi, ry.hi, rz.hi);
                    // Adjacent X intervals with the same Y/Z footprint have the
                    // same union, including runs from several consecutive rows.
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
            }
        }
    }
    if (have_run) occupy(run_lo, run_hi);
}

// Project the exact fine occupied union after its integer boxes have already
// been built for RISP. Positive-volume intersection distributes over that union,
// so this gives the same coarse cells as coarsenOccupiedVolumes without scanning
// every fine bit again. Boxes keep upper-exclusive logical fine coordinates.
// The caller resets cloud occupancy; virtual occupied cells are not listed.
inline void coarsenOccupiedBoxes(
    const std::vector<pathcover_boxes::Box>& boxes,
    const ResolutionRatio& resolution_ratio,
    const Eigen::Vector3i& search_origin_index,
    const Eigen::Vector3i& dimensions,
    std::vector<signed char>& cells,
    std::vector<int>& occupied_indices) {
    auto floorDivide = [](int64_t value, int64_t divisor) {
        return value / divisor - static_cast<int64_t>(value < 0 && value % divisor != 0);
    };
    for (const auto& box : boxes) {
        Eigen::Vector3i lower, upper;
        for (int axis = 0; axis < 3; ++axis) {
            const int64_t fine_lower = static_cast<int64_t>(box.lower[axis]) * resolution_ratio.first;
            const int64_t fine_upper = static_cast<int64_t>(box.upper[axis]) * resolution_ratio.first;
            const int64_t lo = floorDivide(fine_lower, resolution_ratio.second) - search_origin_index[axis];
            const int64_t hi = -floorDivide(-fine_upper, resolution_ratio.second) - 1 - search_origin_index[axis];
            lower[axis] = static_cast<int>(std::clamp<int64_t>(lo, 0, dimensions[axis]));
            upper[axis] = static_cast<int>(std::clamp<int64_t>(hi, -1, dimensions[axis] - 1));
        }
        if ((lower.array() > upper.array()).any()) continue;
        for (int z = lower.z(); z <= upper.z(); ++z) {
            for (int y = lower.y(); y <= upper.y(); ++y) {
                const int row = dimensions.x() * (y + dimensions.y() * z);
                for (int x = lower.x(); x <= upper.x(); ++x) {
                    const int index = row + x;
                    if (cells[index] == 0) {
                        cells[index] = 100;
                        occupied_indices.push_back(index);
                    }
                }
            }
        }
    }
}

} // namespace pathcover_grid
