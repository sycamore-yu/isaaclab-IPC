#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <cstdint>
#include <tuple>
#include <utility>
#include <vector>

#include "occupancy_inflation/voxel_bit_grid.h"

namespace pathcover_boxes {

using Cell = Eigen::Vector3i;

// World-zero voxel indices. Upper is exclusive on every axis. Boxes are
// disjoint as voxel sets; their physical closed AABBs may share boundary faces.
struct Box {
    Cell lower;
    Cell upper;
};

namespace detail {

// The complete key gives the same final order regardless of how each plane's
// disjoint XY rectangles were discovered. Only identical XY footprints merge.
inline std::vector<Box> mergePlanes(std::vector<Box> runs) {
    std::sort(runs.begin(), runs.end(), [](const Box& a, const Box& b) {
        return std::tie(a.lower.x(), a.upper.x(), a.lower.y(), a.upper.y(), a.lower.z())
             < std::tie(b.lower.x(), b.upper.x(), b.lower.y(), b.upper.y(), b.lower.z());
    });
    std::size_t count = 0;
    for (std::size_t i = 0; i < runs.size(); ++i) {
        const Box& current = runs[i];
        if (count && runs[count - 1].lower.x() == current.lower.x()
            && runs[count - 1].upper.x() == current.upper.x()
            && runs[count - 1].lower.y() == current.lower.y()
            && runs[count - 1].upper.y() == current.upper.y()
            && runs[count - 1].upper.z() == current.lower.z()) {
            runs[count - 1].upper.z() = current.upper.z();
        } else {
            if (count != i) runs[count] = current;
            ++count;
        }
    }
    runs.resize(count);
    return runs;
}

// Unordered-cell reference: maximal x runs with one-cell y/z thickness are
// sorted by interval before merging consecutive y rows.
inline std::vector<Box> mergeRuns(std::vector<Box> runs) {
    std::sort(runs.begin(), runs.end(), [](const Box& a, const Box& b) {
        return std::tie(a.lower.z(), a.lower.x(), a.upper.x(), a.lower.y())
             < std::tie(b.lower.z(), b.lower.x(), b.upper.x(), b.lower.y());
    });
    std::size_t count = 0;
    for (std::size_t i = 0; i < runs.size(); ++i) {
        const Box& current = runs[i];
        if (count && runs[count - 1].lower.z() == current.lower.z()
            && runs[count - 1].lower.x() == current.lower.x()
            && runs[count - 1].upper.x() == current.upper.x()
            && runs[count - 1].upper.y() == current.lower.y()) {
            runs[count - 1].upper.y() = current.upper.y();
        } else {
            if (count != i) runs[count] = current;
            ++count;
        }
    }
    runs.resize(count);
    return mergePlanes(std::move(runs));
}

} // namespace detail

// Unordered cells, including duplicates. Pass an rvalue to avoid copying the
// input before sorting. The upper-exclusive integer coordinates must fit Cell.
inline std::vector<Box> compress(std::vector<Cell> occupied) {
    std::sort(occupied.begin(), occupied.end(), [](const Cell& a, const Cell& b) {
        return std::tie(a.z(), a.y(), a.x()) < std::tie(b.z(), b.y(), b.x());
    });
    std::vector<Box> runs;
    for (const Cell& cell : occupied) {
        if (!runs.empty() && runs.back().lower.y() == cell.y()
            && runs.back().lower.z() == cell.z() && cell.x() <= runs.back().upper.x()) {
            runs.back().upper.x() = std::max(runs.back().upper.x(), cell.x() + 1);
        } else {
            runs.push_back({cell, cell + Cell::Ones()});
        }
    }
    return detail::mergeRuns(std::move(runs));
}

// Production path: scan packed words directly, without materializing or
// sorting one Cell per occupied voxel. Consecutive runs cross word boundaries.
// Rows already arrive in increasing x order. Match equal intervals against the
// immediately preceding row, avoiding the first global sort of all x runs.
inline std::vector<Box> compress(const occupancy_inflation::VoxelBitGrid& grid) {
    if (grid.empty()) return {};
    const auto& window = grid.window();
    std::vector<Box> rectangles;
    std::vector<std::size_t> previous_row, current_row;
    for (int z = 0; z < window.dims.z(); ++z) {
        previous_row.clear();
        for (int y = 0; y < window.dims.y(); ++y) {
            current_row.clear();
            std::size_t previous = 0;
            const auto* row = grid.data() + window.rowOffset(y, z);
            int pending_lower = -1, pending_upper = -1;
            auto flush = [&]() {
                if (pending_lower >= 0) {
                    const Cell lower(window.base.x() + pending_lower,
                                     window.base.y() + y, window.base.z() + z);
                    const Cell upper(window.base.x() + pending_upper,
                                     window.base.y() + y + 1, window.base.z() + z + 1);
                    while (previous < previous_row.size()
                        && rectangles[previous_row[previous]].lower.x() < lower.x()) {
                        ++previous;
                    }
                    std::size_t index;
                    if (previous < previous_row.size()
                        && rectangles[previous_row[previous]].lower.x() == lower.x()
                        && rectangles[previous_row[previous]].upper.x() == upper.x()) {
                        index = previous_row[previous];
                        rectangles[index].upper.y() = upper.y();
                    } else {
                        index = rectangles.size();
                        rectangles.push_back({lower, upper});
                    }
                    current_row.push_back(index);
                }
            };
            for (std::size_t word = 0; word < window.wordsPerRow(); ++word) {
                uint64_t bits = row[word];
                const int offset = static_cast<int>(word * occupancy_inflation::kWordBits);
                const int valid = std::min(occupancy_inflation::kWordBits, window.dims.x() - offset);
                if (valid < occupancy_inflation::kWordBits) bits &= (uint64_t{1} << valid) - 1;
                while (bits) {
                    const int first = __builtin_ctzll(bits);
                    const uint64_t shifted = bits >> first;
                    const int length = ~shifted ? __builtin_ctzll(~shifted) : 64;
                    const int last = first + length;
                    const int lower = offset + first, upper = offset + last;
                    if (pending_lower >= 0 && pending_upper == lower) {
                        pending_upper = upper;
                    } else {
                        flush();
                        pending_lower = lower;
                        pending_upper = upper;
                    }
                    bits = last == 64 ? 0 : bits & (~uint64_t{0} << last);
                }
            }
            flush();
            previous_row.swap(current_row);
        }
    }
    return detail::mergePlanes(std::move(rectangles));
}

} // namespace pathcover_boxes
