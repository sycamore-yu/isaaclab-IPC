#include "pathcover_grid_cuda.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cstring>
#include <stdexcept>
#include <string>

namespace pathcover_cuda {
namespace {

void checked(cudaError_t code, const char* operation) {
    if (code != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(code));
    }
}

template<class T> struct DeviceBuffer {
    T* data = nullptr;
    std::size_t capacity = 0;
    ~DeviceBuffer() { if (data) cudaFree(data); }
    void ensure(std::size_t count) {
        if (capacity >= count) return;
        T* fresh = nullptr;
        checked(cudaMalloc(reinterpret_cast<void**>(&fresh), count * sizeof(T)), "allocate grid projection buffer");
        if (data) cudaFree(data);
        data = fresh;
        capacity = count;
    }
};

template<class T> struct PinnedBuffer {
    T* data = nullptr;
    std::size_t capacity = 0;
    ~PinnedBuffer() { if (data) cudaFreeHost(data); }
    void ensure(std::size_t count) {
        if (capacity >= count) return;
        T* fresh = nullptr;
        checked(cudaMallocHost(reinterpret_cast<void**>(&fresh), count * sizeof(T)), "allocate pinned grid projection buffer");
        if (data) cudaFreeHost(data);
        data = fresh;
        capacity = count;
    }
};

std::int64_t floorDivide(std::int64_t value, std::int64_t divisor) {
    return value / divisor - static_cast<std::int64_t>(value < 0 && value % divisor != 0);
}

__global__ void project_cells(const std::uint64_t* fine, const int2* ranges,
                             int nx, int ny, int count, int fine_ny,
                             std::size_t words_per_row, signed char* output) {
    const unsigned int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= static_cast<unsigned int>(count)) return;
    const int x = cell % nx;
    const int y = (cell / nx) % ny;
    const int z = cell / (nx * ny);
    const int2 xr = ranges[x], yr = ranges[nx + y], zr = ranges[nx + ny + z];
    if (xr.x > xr.y || yr.x > yr.y || zr.x > zr.y) {
        output[cell] = 0;
        return;
    }
    const int first_word = xr.x / 64, last_word = xr.y / 64;
    const std::uint64_t lower_mask = ~std::uint64_t{0} << (xr.x % 64);
    const int high = xr.y % 64;
    const std::uint64_t upper_mask = high == 63 ? ~std::uint64_t{0} : (std::uint64_t{1} << (high + 1)) - 1;
    for (int iz = zr.x; iz <= zr.y; ++iz) {
        for (int iy = yr.x; iy <= yr.y; ++iy) {
            const auto* row = fine + (std::size_t(iz) * fine_ny + iy) * words_per_row;
            for (int word = first_word; word <= last_word; ++word) {
                std::uint64_t bits = row[word];
                if (word == first_word) bits &= lower_mask;
                if (word == last_word) bits &= upper_mask;
                if (bits) {
                    output[cell] = 100;
                    return;
                }
            }
        }
    }
    output[cell] = 0;
}

}  // namespace

struct GridProjector::Impl {
    cudaStream_t stream{};
    DeviceBuffer<std::uint64_t> bits;
    DeviceBuffer<int2> ranges;
    DeviceBuffer<signed char> cells;
    PinnedBuffer<std::uint64_t> host_bits;
    PinnedBuffer<int2> host_ranges;
    PinnedBuffer<signed char> host_cells;

    Impl() { checked(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create grid projection stream"); }
    ~Impl() { if (stream) cudaStreamDestroy(stream); }

    std::vector<signed char> project(
        const occupancy_inflation::VoxelBitGrid& fine,
        const std::pair<std::int64_t, std::int64_t>& ratio,
        const Eigen::Vector3i& origin, const Eigen::Vector3i& dimensions) {
        if ((dimensions.array() <= 0).any() || ratio.first <= 0 || ratio.second < ratio.first) {
            throw std::invalid_argument("Coarse projection requires positive dimensions and 0 < fine/search <= 1");
        }
        if (dimensions.x() > INT_MAX / dimensions.y()
            || dimensions.x() * dimensions.y() > INT_MAX / dimensions.z()) {
            throw std::invalid_argument("Coarse projection exceeds integer cell indexing");
        }
        const int count = dimensions.x() * dimensions.y() * dimensions.z();
        std::vector<signed char> output(count, 0);
        if (fine.empty()) return output;
        const auto& window = fine.window();
        const std::size_t words = fine.wordCount();
        const std::size_t range_count = std::size_t(dimensions.x()) + dimensions.y() + dimensions.z();
        bits.ensure(words); ranges.ensure(range_count); cells.ensure(count);
        host_bits.ensure(words); host_ranges.ensure(range_count); host_cells.ensure(count);
        std::size_t offset = 0;
        for (int axis = 0; axis < 3; ++axis) {
            const std::int64_t first = window.base[axis];
            const std::int64_t last = first + window.dims[axis] - 1;
            for (int index = 0; index < dimensions[axis]; ++index) {
                const std::int64_t coarse = std::int64_t(origin[axis]) + index;
                // Integer floor(lower) and ceil(upper)-1 include precisely the
                // fine cells with positive-volume overlap, including negatives.
                const auto lower = std::max(first, floorDivide(coarse * ratio.second, ratio.first));
                const auto upper = std::min(last, -floorDivide(-(coarse + 1) * ratio.second, ratio.first) - 1);
                host_ranges.data[offset++] = lower > upper ? make_int2(0, -1)
                    : make_int2(static_cast<int>(lower - first), static_cast<int>(upper - first));
            }
        }
        std::memcpy(host_bits.data, fine.data(), words * sizeof(std::uint64_t));
        checked(cudaMemcpyAsync(bits.data, host_bits.data, words * sizeof(std::uint64_t), cudaMemcpyHostToDevice, stream), "upload fine projection grid");
        checked(cudaMemcpyAsync(ranges.data, host_ranges.data, range_count * sizeof(int2), cudaMemcpyHostToDevice, stream), "upload integer projection ranges");
        project_cells<<<(static_cast<unsigned int>(count) + 255) / 256, 256, 0, stream>>>(
            bits.data, ranges.data, dimensions.x(), dimensions.y(), count,
            window.dims.y(), window.wordsPerRow(), cells.data);
        checked(cudaGetLastError(), "project fine occupied volumes");
        checked(cudaMemcpyAsync(host_cells.data, cells.data, count * sizeof(signed char), cudaMemcpyDeviceToHost, stream), "download coarse occupancy");
        checked(cudaStreamSynchronize(stream), "wait for coarse occupancy");
        std::memcpy(output.data(), host_cells.data, count * sizeof(signed char));
        return output;
    }
};

GridProjector::GridProjector() : impl_(std::make_unique<Impl>()) {}
GridProjector::~GridProjector() = default;
std::vector<signed char> GridProjector::project(
    const occupancy_inflation::VoxelBitGrid& fine,
    const std::pair<std::int64_t, std::int64_t>& ratio,
    const Eigen::Vector3i& origin, const Eigen::Vector3i& dimensions) {
    return impl_->project(fine, ratio, origin, dimensions);
}

}  // namespace pathcover_cuda
