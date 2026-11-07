#include "pathcover_boxes_cuda.h"

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <cuda/std/tuple>
#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cstring>
#include <stdexcept>
#include <string>

namespace pathcover_cuda {
namespace {

void checked(cudaError_t code, const char* operation) {
    if (code != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(code));
}

template <class T> struct DeviceBuffer {
    T* ptr = nullptr;
    std::size_t capacity = 0;
    ~DeviceBuffer() { if (ptr) cudaFree(ptr); }
    void ensure(std::size_t n) {
        if (capacity >= n) return;
        T* fresh = nullptr;
        checked(cudaMalloc(reinterpret_cast<void**>(&fresh), n * sizeof(T)), "cudaMalloc");
        if (ptr) cudaFree(ptr);
        ptr = fresh;
        capacity = n;
    }
};

template <class T> struct PinnedBuffer {
    T* ptr = nullptr;
    std::size_t capacity = 0;
    ~PinnedBuffer() { if (ptr) cudaFreeHost(ptr); }
    void ensure(std::size_t n) {
        if (capacity >= n) return;
        T* fresh = nullptr;
        checked(cudaMallocHost(reinterpret_cast<void**>(&fresh), n * sizeof(T)), "cudaMallocHost");
        if (ptr) cudaFreeHost(ptr);
        ptr = fresh;
        capacity = n;
    }
};

struct IntegerBox { int lx, ly, lz, ux, uy, uz; };
static_assert(sizeof(IntegerBox) == 24);
struct Counts { std::uint32_t starts, ends; };
static_assert(sizeof(Counts) == 8);
struct AddCounts {
    __host__ __device__ Counts operator()(Counts a, Counts b) const {
        return {a.starts + b.starts, a.ends + b.ends};
    }
};
struct Window { int bx, by, bz, nx, ny, nz; std::size_t words_per_row; };
struct RunKey {
    __host__ __device__ auto operator()(IntegerBox& b) const {
        return cuda::std::tie(b.lz, b.lx, b.ux, b.ly);
    }
};
struct PlaneKey {
    __host__ __device__ auto operator()(IntegerBox& b) const {
        return cuda::std::tie(b.lx, b.ux, b.ly, b.uy, b.lz);
    }
};

__device__ std::uint64_t valid_word(const std::uint64_t* bits, std::size_t index, Window w) {
    const int x = static_cast<int>(index % w.words_per_row) * 64;
    const int valid = min(64, w.nx - x);
    const std::uint64_t word = bits[index];
    return valid == 64 ? word : word & ((std::uint64_t{1} << valid) - 1);
}

__device__ void boundaries(const std::uint64_t* bits, std::size_t index, Window w,
                           std::uint64_t& starts, std::uint64_t& ends) {
    const std::size_t word_x = index % w.words_per_row;
    const std::uint64_t word = valid_word(bits, index, w);
    const std::uint64_t previous = word_x ? bits[index - 1] >> 63 : 0;
    const std::uint64_t next = word_x + 1 < w.words_per_row ? (bits[index + 1] & 1) : 0;
    starts = word & ~((word << 1) | previous);
    ends = word & ~((word >> 1) | (next << 63));
}

__global__ void count_words(const std::uint64_t* bits, Counts* counts, std::size_t words, Window w) {
    const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= words) return;
    std::uint64_t starts, ends;
    boundaries(bits, i, w, starts, ends);
    counts[i] = {static_cast<std::uint32_t>(__popcll(starts)), static_cast<std::uint32_t>(__popcll(ends))};
}

__global__ void total_counts(const Counts* counts, const Counts* prefix, std::size_t words, Counts* total) {
    if (threadIdx.x == 0 && blockIdx.x == 0) *total = AddCounts{}(prefix[words - 1], counts[words - 1]);
}

__global__ void emit_runs(const std::uint64_t* bits, const Counts* prefix, std::size_t words,
                          Window w, IntegerBox* output) {
    const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= words) return;
    std::uint64_t starts, ends;
    boundaries(bits, i, w, starts, ends);
    const int x = static_cast<int>(i % w.words_per_row) * 64;
    const std::size_t row = i / w.words_per_row;
    const int y = static_cast<int>(row % w.ny), z = static_cast<int>(row / w.ny);
    std::uint32_t start_index = prefix[i].starts, end_index = prefix[i].ends;
    while (starts) {
        const int bit = __ffsll(static_cast<long long>(starts)) - 1;
        IntegerBox& box = output[start_index++];
        box.lx = w.bx + x + bit;
        box.ly = w.by + y;
        box.lz = w.bz + z;
        box.uy = w.by + y + 1;
        box.uz = w.bz + z + 1;
        starts &= starts - 1;
    }
    while (ends) {
        const int bit = __ffsll(static_cast<long long>(ends)) - 1;
        output[end_index++].ux = w.bx + x + bit + 1;
        ends &= ends - 1;
    }
}

__global__ void y_heads(const IntegerBox* runs, std::uint32_t* flags, std::uint32_t n) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    bool head = i == 0;
    if (i) {
        const IntegerBox a = runs[i - 1], b = runs[i];
        head = a.lz != b.lz || a.lx != b.lx || a.ux != b.ux || a.uy != b.ly;
    }
    flags[i] = head;
}

__global__ void fill_sentinels(IntegerBox* output, std::uint32_t n) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = {INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX};
}

__global__ void emit_rectangles(const IntegerBox* runs, const std::uint32_t* flags,
                                const std::uint32_t* groups, std::uint32_t n,
                                IntegerBox* output, Counts* total) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const std::uint32_t dst = groups[i] - 1;
    const IntegerBox box = runs[i];
    if (flags[i]) {
        // Heads and tails write disjoint integer fields, including one-run groups.
        output[dst].lx = box.lx;
        output[dst].ly = box.ly;
        output[dst].lz = box.lz;
        output[dst].ux = box.ux;
        output[dst].uz = box.uz;
    }
    if (i + 1 == n || flags[i + 1]) output[dst].uy = box.uy;
    if (i + 1 == n) *total = {groups[i], n};
}

unsigned int blocks(std::size_t n) { return static_cast<unsigned int>((n + 255) / 256); }

// This is the same linear fold as V21 detail::mergePlanes, after its sort.
// The independent CPU production implementation remains the benchmark oracle.
std::vector<pathcover_boxes::Box> finish_z(const IntegerBox* rectangles, std::size_t count) {
    std::vector<pathcover_boxes::Box> boxes;
    boxes.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        const IntegerBox& b = rectangles[i];
        if (!boxes.empty() && boxes.back().lower.x() == b.lx && boxes.back().upper.x() == b.ux
            && boxes.back().lower.y() == b.ly && boxes.back().upper.y() == b.uy
            && boxes.back().upper.z() == b.lz) {
            boxes.back().upper.z() = b.uz;
        } else {
            boxes.push_back({pathcover_boxes::Cell(b.lx, b.ly, b.lz), pathcover_boxes::Cell(b.ux, b.uy, b.uz)});
        }
    }
    return boxes;
}

}  // namespace

struct BoxCompressor::Impl {
    cudaStream_t stream{};
    DeviceBuffer<std::uint64_t> bits;
    DeviceBuffer<Counts> counts, prefix, total;
    DeviceBuffer<IntegerBox> first, second;
    DeviceBuffer<std::uint32_t> flags, groups;
    DeviceBuffer<std::uint8_t> temporary;
    PinnedBuffer<std::uint64_t> host_bits;
    PinnedBuffer<IntegerBox> host_boxes;
    PinnedBuffer<Counts> host_total;

    Impl() { checked(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create compressor stream"); }
    ~Impl() { if (stream) cudaStreamDestroy(stream); }

    std::vector<pathcover_boxes::Box> compress(const occupancy_inflation::VoxelBitGrid& grid) {
        if (grid.empty()) return {};
        const auto& win = grid.window();
        const Window w{win.base.x(), win.base.y(), win.base.z(), win.dims.x(), win.dims.y(), win.dims.z(), win.wordsPerRow()};
        const std::size_t words = grid.wordCount();
        bits.ensure(words); counts.ensure(words); prefix.ensure(words); total.ensure(1);
        host_bits.ensure(words); host_total.ensure(1);
        std::memcpy(host_bits.ptr, grid.data(), words * sizeof(std::uint64_t));
        checked(cudaMemcpyAsync(bits.ptr, host_bits.ptr, words * sizeof(std::uint64_t), cudaMemcpyHostToDevice, stream), "upload cumulative bit-grid");
        count_words<<<blocks(words), 256, 0, stream>>>(bits.ptr, counts.ptr, words, w);
        checked(cudaGetLastError(), "count X-run boundaries");
        std::size_t scan_bytes = 0;
        checked(cub::DeviceScan::ExclusiveScan(nullptr, scan_bytes, counts.ptr, prefix.ptr, AddCounts{}, Counts{0, 0}, words, stream), "size word scan");
        temporary.ensure(scan_bytes);
        checked(cub::DeviceScan::ExclusiveScan(temporary.ptr, scan_bytes, counts.ptr, prefix.ptr, AddCounts{}, Counts{0, 0}, words, stream), "scan X-run boundaries");
        total_counts<<<1, 1, 0, stream>>>(counts.ptr, prefix.ptr, words, total.ptr);
        checked(cudaGetLastError(), "finish X-run count");
        checked(cudaMemcpyAsync(host_total.ptr, total.ptr, sizeof(Counts), cudaMemcpyDeviceToHost, stream), "read X-run count");
        checked(cudaStreamSynchronize(stream), "wait for X-run count");
        const std::uint32_t runs = host_total.ptr->starts;
        if (runs != host_total.ptr->ends || runs > static_cast<std::uint32_t>(INT_MAX)) {
            throw std::runtime_error("X-run count mismatch or CUB item-count range exceeded");
        }
        if (!runs) return {};
        first.ensure(runs); second.ensure(runs); flags.ensure(runs); groups.ensure(runs); host_boxes.ensure(runs);
        emit_runs<<<blocks(words), 256, 0, stream>>>(bits.ptr, prefix.ptr, words, w, first.ptr);
        checked(cudaGetLastError(), "emit exact X-runs");
        std::size_t run_sort_bytes = 0, plane_sort_bytes = 0, group_scan_bytes = 0;
        checked(cub::DeviceRadixSort::SortKeys(nullptr, run_sort_bytes, first.ptr, second.ptr, runs, RunKey{}, 0, 128, stream), "size run tuple sort");
        checked(cub::DeviceRadixSort::SortKeys(nullptr, plane_sort_bytes, first.ptr, second.ptr, runs, PlaneKey{}, 0, 160, stream), "size plane tuple sort");
        checked(cub::DeviceScan::InclusiveSum(nullptr, group_scan_bytes, flags.ptr, groups.ptr, runs, stream), "size Y group scan");
        temporary.ensure(std::max({run_sort_bytes, plane_sort_bytes, group_scan_bytes}));
        checked(cub::DeviceRadixSort::SortKeys(temporary.ptr, run_sort_bytes, first.ptr, second.ptr, runs, RunKey{}, 0, 128, stream), "sort exact X-runs");
        y_heads<<<blocks(runs), 256, 0, stream>>>(second.ptr, flags.ptr, runs);
        checked(cudaGetLastError(), "mark consecutive Y groups");
        checked(cub::DeviceScan::InclusiveSum(temporary.ptr, group_scan_bytes, flags.ptr, groups.ptr, runs, stream), "scan consecutive Y groups");
        fill_sentinels<<<blocks(runs), 256, 0, stream>>>(first.ptr, runs);
        emit_rectangles<<<blocks(runs), 256, 0, stream>>>(second.ptr, flags.ptr, groups.ptr, runs, first.ptr, total.ptr);
        checked(cudaGetLastError(), "emit exact XY rectangles");
        checked(cub::DeviceRadixSort::SortKeys(temporary.ptr, plane_sort_bytes, first.ptr, second.ptr, runs, PlaneKey{}, 0, 160, stream), "sort V21 plane keys");
        checked(cudaMemcpyAsync(host_boxes.ptr, second.ptr, runs * sizeof(IntegerBox), cudaMemcpyDeviceToHost, stream), "download sorted XY rectangles");
        checked(cudaMemcpyAsync(host_total.ptr, total.ptr, sizeof(Counts), cudaMemcpyDeviceToHost, stream), "download XY rectangle count");
        checked(cudaStreamSynchronize(stream), "wait for sorted rectangles");
        const std::uint32_t rectangles = host_total.ptr->starts;
        if (!rectangles || rectangles > runs || host_total.ptr->ends != runs) throw std::runtime_error("XY rectangle count mismatch");
        auto output = finish_z(host_boxes.ptr, rectangles);
        return output;
    }
};

BoxCompressor::BoxCompressor() : impl_(std::make_unique<Impl>()) {}
BoxCompressor::~BoxCompressor() = default;
std::vector<pathcover_boxes::Box> BoxCompressor::compress(const occupancy_inflation::VoxelBitGrid& grid) { return impl_->compress(grid); }

}  // namespace pathcover_cuda
