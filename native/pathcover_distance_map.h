// Exact integer distance-potential dilation for the upstream DMP search.
#pragma once
#include <array>
#include <cstdint>
#include <memory>
#include <vector>

namespace pathcover_cuda {
class DistanceMap {
    struct Workspace;
    std::unique_ptr<Workspace> workspace_;
public:
    DistanceMap();
    ~DistanceMap();
    std::vector<int8_t> compute(const std::vector<int8_t>& occupancy,
        const std::array<int, 3>& dimensions, const std::vector<std::array<int, 4>>& mask);
};
}
