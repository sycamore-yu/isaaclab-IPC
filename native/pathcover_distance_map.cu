#include "pathcover_distance_map.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <stdexcept>
#include <string>

namespace pathcover_cuda {
namespace {
void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(
        std::string("PathCover CUDA distance map: ") + cudaGetErrorString(status));
}
constexpr uint32_t requiredFreeMask(int dx, int dy, int dz) {
    uint32_t mask = 0;
    for (int sx = 0; sx < 2; ++sx)
        for (int sy = 0; sy < 2; ++sy)
            for (int sz = 0; sz < 2; ++sz) {
                const int x = sx * dx, y = sy * dy, z = sz * dz;
                if (x || y || z)
                    mask |= uint32_t{1} << ((x + 1) * 9 + (y + 1) * 3 + z + 1);
            }
    return mask;
}
struct RequiredMasks { uint32_t values[27]; };
constexpr RequiredMasks makeRequiredMasks() {
    RequiredMasks result{};
    for (int index = 0; index < 27; ++index)
        result.values[index] = requiredFreeMask(index / 9 - 1, index / 3 % 3 - 1, index % 3 - 1);
    return result;
}
__device__ __constant__ RequiredMasks required_masks = makeRequiredMasks();

__global__ void computeLegalMoves(const int8_t* occupancy, uint32_t* output,
                                  int count, int nx, int ny, int nz) {
    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < count; id += gridDim.x * blockDim.x) {
        // initialize() maps every positive occupancy to 100. Unknown/free values
        // remain below 100, matching GraphSearch::isFree. Virtual obstacles are
        // already part of occupancy; potential costs and search state are absent.
        if (occupancy[id] > 0) {
            output[id] = 0;
            continue;
        }
        const int x = id % nx, y = id / nx % ny, z = id / (nx * ny);
        uint32_t free_mask = 0;
        #pragma unroll
        for (int index = 0; index < 27; ++index) {
            if (index == 13) continue;
            const int dx = index / 9 - 1, dy = index / 3 % 3 - 1, dz = index % 3 - 1;
            const int xx = x + dx, yy = y + dy, zz = z + dz;
            if (xx >= 0 && xx < nx && yy >= 0 && yy < ny && zz >= 0 && zz < nz
                    && occupancy[id + dx + nx * (dy + ny * dz)] <= 0)
                free_mask |= uint32_t{1} << index;
        }
        uint32_t legal = 0;
        #pragma unroll
        for (int index = 0; index < 27; ++index) {
            if (index == 13) continue;
            const uint32_t required = required_masks.values[index];
            if ((free_mask & required) == required) legal |= uint32_t{1} << index;
        }
        output[id] = legal;
    }
}
__global__ void initialize(const int8_t* input, int* values, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += gridDim.x * blockDim.x)
        values[i] = input[i] > 0 ? 100 : input[i];
}
__global__ void dilate(const int* occupied, int count, const int4* mask, int mask_count,
                      int* values, int nx, int ny, int nz) {
    const std::size_t pairs = static_cast<std::size_t>(count) * mask_count;
    for (std::size_t pair = blockIdx.x * blockDim.x + threadIdx.x;
         pair < pairs; pair += gridDim.x * blockDim.x) {
        const int id = occupied[pair / mask_count];
        const int4 offset = mask[pair % mask_count];
        const int x = id % nx + offset.x;
        const int y = id / nx % ny + offset.y;
        const int z = id / (nx * ny) + offset.z;
        if (x >= 0 && x < nx && y >= 0 && y < ny && z >= 0 && z < nz)
            atomicMax(values + x + nx * (y + ny * z), offset.w);
    }
}
__global__ void pack(const int* values, int8_t* output, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += gridDim.x * blockDim.x)
        output[i] = static_cast<int8_t>(values[i]);
}
}

struct DistanceMap::Workspace {
    cudaStream_t stream = nullptr;
    int8_t* bytes = nullptr;
    int* values = nullptr;
    int* occupied = nullptr;
    int4* mask = nullptr;
    uint32_t* legal_moves = nullptr;
    std::size_t capacity = 0;
    std::vector<std::array<int, 4>> host_mask;
    Workspace() { check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)); }
    ~Workspace() {
        cudaFree(bytes); cudaFree(values); cudaFree(occupied); cudaFree(mask); cudaFree(legal_moves);
        cudaStreamDestroy(stream);
    }
    void reserve(std::size_t size, const std::vector<std::array<int, 4>>& offsets) {
        if (size > capacity) {
            cudaFree(bytes); cudaFree(values); cudaFree(occupied); cudaFree(legal_moves);
            bytes = nullptr; values = nullptr; occupied = nullptr;
            legal_moves = nullptr;
            check(cudaMalloc(&bytes, size * sizeof(int8_t)));
            check(cudaMalloc(&values, size * sizeof(int)));
            check(cudaMalloc(&occupied, size * sizeof(int)));
            check(cudaMalloc(&legal_moves, size * sizeof(uint32_t)));
            capacity = size;
        }
        if (host_mask != offsets) {
            cudaFree(mask); mask = nullptr;
            if (!offsets.empty()) {
                check(cudaMalloc(&mask, offsets.size() * sizeof(int4)));
                check(cudaMemcpyAsync(mask, offsets.data(), offsets.size() * sizeof(int4),
                                      cudaMemcpyHostToDevice, stream));
            }
            host_mask = offsets;
        }
    }
};

DistanceMap::DistanceMap() : workspace_(std::make_unique<Workspace>()) {}
DistanceMap::~DistanceMap() = default;
DistanceMapResult DistanceMap::compute(const std::vector<int8_t>& occupancy,
        const std::array<int, 3>& dimensions, const std::vector<std::array<int, 4>>& mask) {
    auto& work = *workspace_;
    work.reserve(occupancy.size(), mask);
    std::vector<int> occupied;
    occupied.reserve(occupancy.size() / 8);
    for (std::size_t i = 0; i < occupancy.size(); ++i)
        if (occupancy[i] > 0) occupied.push_back(i);
    check(cudaMemcpyAsync(work.bytes, occupancy.data(), occupancy.size() * sizeof(int8_t),
                          cudaMemcpyHostToDevice, work.stream));
    if (!occupied.empty()) check(cudaMemcpyAsync(work.occupied, occupied.data(),
        occupied.size() * sizeof(int), cudaMemcpyHostToDevice, work.stream));
    const int blocks = std::min<std::size_t>((occupancy.size() + 255) / 256, 4096);
    initialize<<<blocks, 256, 0, work.stream>>>(work.bytes, work.values, occupancy.size());
    computeLegalMoves<<<blocks, 256, 0, work.stream>>>(work.bytes, work.legal_moves,
        occupancy.size(), dimensions[0], dimensions[1], dimensions[2]);
    if (!occupied.empty() && !mask.empty()) {
        const int dilation_blocks = std::min<std::size_t>((occupied.size() * mask.size() + 255) / 256, 16384);
        dilate<<<dilation_blocks, 256, 0, work.stream>>>(work.occupied, occupied.size(), work.mask,
            mask.size(), work.values, dimensions[0], dimensions[1], dimensions[2]);
    }
    pack<<<blocks, 256, 0, work.stream>>>(work.values, work.bytes, occupancy.size());
    check(cudaGetLastError());
    DistanceMapResult output{std::vector<int8_t>(occupancy.size()), std::vector<uint32_t>(occupancy.size())};
    check(cudaMemcpyAsync(output.potential.data(), work.bytes, output.potential.size() * sizeof(int8_t),
                          cudaMemcpyDeviceToHost, work.stream));
    check(cudaMemcpyAsync(output.legal_moves.data(), work.legal_moves, output.legal_moves.size() * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost, work.stream));
    check(cudaStreamSynchronize(work.stream));
    return output;
}
}
