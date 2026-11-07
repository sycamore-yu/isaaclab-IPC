// Exact integer distance-potential dilation for the upstream DMP search.
#pragma once
#include <array>
#include <cstdint>
#include <memory>
#include <vector>

namespace pathcover_cuda {
// Native radial-potential offsets stay below 100 except at the occupied centre.
// The potential and legal moves describe the same immutable coarse occupancy.
// Direction bit: (dx + 1) * 9 + (dy + 1) * 3 + dz + 1; centre bit 13 is unused.
struct DistanceMapResult {
    std::vector<int8_t> potential;
    std::vector<uint32_t> legal_moves;
};
class DistanceMap {
    struct Workspace;
    std::unique_ptr<Workspace> workspace_;
public:
    DistanceMap();
    ~DistanceMap();
    DistanceMapResult compute(const std::vector<int8_t>& occupancy,
        const std::array<int, 3>& dimensions, const std::vector<std::array<int, 4>>& mask);
};
}
