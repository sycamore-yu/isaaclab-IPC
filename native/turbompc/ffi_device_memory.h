#pragma once

// PyTorch allocator adapter for the unmodified TurboMPC CUDA kernels.
// The upstream type names are retained; this header has no XLA dependency.
#include <c10/cuda/CUDACachingAllocator.h>
#include <cuda_runtime.h>
#include <cudss.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace turbompc {

struct FfiDeviceAllocator {
  cudaStream_t stream = nullptr;
};

inline size_t AllocationSize(size_t bytes) { return std::max<size_t>(bytes, 1); }

template <typename T>
inline void FfiAllocate(FfiDeviceAllocator& allocator, T** pointer, size_t bytes) {
  *pointer = static_cast<T*>(
      c10::cuda::CUDACachingAllocator::raw_alloc_with_stream(
          AllocationSize(bytes), allocator.stream));
}

inline void FfiFree(FfiDeviceAllocator&, void* pointer, size_t) {
  if (pointer) c10::cuda::CUDACachingAllocator::raw_delete(pointer);
}

inline void ReportAllocatorError(const char* text) {
  std::fprintf(stderr, "TurboMPC Torch allocator: %s\n", text);
}

inline cudssStatus_t SetCudssDeviceMemory(cudssHandle_t handle,
                                        FfiDeviceAllocator& allocator) {
  cudssDeviceMemHandler_t handler{};
  handler.ctx = &allocator;
  std::strcpy(handler.name, "TurboMPC Torch allocator");
  handler.device_alloc = [](void* context, void** pointer, size_t size,
                            cudaStream_t stream) -> int {
    try {
      if (size == 0) { *pointer = nullptr; return 0; }
      auto current = *static_cast<FfiDeviceAllocator*>(context);
      current.stream = stream;
      FfiAllocate(current, pointer, size);
      return 0;
    } catch (const std::exception& error) {
      ReportAllocatorError(error.what());
      return 1;
    }
  };
  handler.device_free = [](void* context, void* pointer, size_t size,
                           cudaStream_t stream) -> int {
    if (!pointer) return 0;
    // Preserve the upstream allocator's ordering before releasing cuDSS memory.
    const auto status = cudaStreamSynchronize(stream);
    if (status != cudaSuccess) {
      ReportAllocatorError(cudaGetErrorString(status));
      return 1;
    }
    try {
      FfiFree(*static_cast<FfiDeviceAllocator*>(context), pointer, size);
      return 0;
    } catch (const std::exception& error) {
      ReportAllocatorError(error.what());
      return 1;
    }
  };
  return cudssSetDeviceMemHandler(handle, &handler);
}
}  // namespace turbompc
