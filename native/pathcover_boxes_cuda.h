// Exact GPU compression of the cumulative packed fine grid.
#pragma once

#include <memory>
#include <vector>

#include "pathcover_boxes.h"

namespace pathcover_cuda {
// Instance-owned stream and reusable storage. The caller selects this backend
// with use_cuda and cuda_min_grid_voxels. CUDA failures propagate as exceptions.
// Output coordinates and ordering match pathcover_boxes::compress(grid).
class BoxCompressor {
public:
    BoxCompressor();
    ~BoxCompressor();
    BoxCompressor(const BoxCompressor&) = delete;
    BoxCompressor& operator=(const BoxCompressor&) = delete;
    std::vector<pathcover_boxes::Box> compress(const occupancy_inflation::VoxelBitGrid& grid);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}  // namespace pathcover_cuda
