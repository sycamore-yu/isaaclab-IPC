// Exact projection from fine occupied volumes to the coarse search grid.
#pragma once

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "occupancy_inflation/voxel_bit_grid.h"

namespace pathcover_cuda {

// One thread reads the fine cells that have positive-volume intersection with
// one coarse cell. No box construction, sorting or device atomics are needed.
// The returned bytes contain cloud occupancy only; the caller keeps workspace
// boundary cells and records the newly occupied indices as before.
class GridProjector {
public:
    GridProjector();
    ~GridProjector();
    GridProjector(const GridProjector&) = delete;
    GridProjector& operator=(const GridProjector&) = delete;
    std::vector<signed char> project(
        const occupancy_inflation::VoxelBitGrid& fine,
        const std::pair<std::int64_t, std::int64_t>& fine_to_search_ratio,
        const Eigen::Vector3i& search_origin_index,
        const Eigen::Vector3i& search_dimensions);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace pathcover_cuda
