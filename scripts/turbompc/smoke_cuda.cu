// Minimal build/runtime probe, not a TurboMPC implementation.
// cuDSS 1.0 API: https://docs.nvidia.com/cuda/cudss/1.0.0/functions.html
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cudss.h>

#include <cstdint>
#include <tuple>
#include <vector>

namespace {
void check(cudssStatus_t status, const char* operation) {
    TORCH_CHECK(status == CUDSS_STATUS_SUCCESS,
                operation, " returned cuDSS status ", static_cast<int>(status));
}

// Keep cleanup valid if any checked cuDSS call throws.
struct SparseObjects {
    cudssHandle_t handle = nullptr;
    cudssConfig_t config = nullptr;
    cudssData_t data = nullptr;
    cudssMatrix_t matrix = nullptr, solution = nullptr, rhs = nullptr;
    ~SparseObjects() {
        if (data) cudssDataDestroy(handle, data);
        if (matrix) cudssMatrixDestroy(matrix);
        if (solution) cudssMatrixDestroy(solution);
        if (rhs) cudssMatrixDestroy(rhs);
        if (config) cudssConfigDestroy(config);
        if (handle) cudssDestroy(handle);
    }
};

__global__ void affine_kernel(const double* input, double* output, int64_t size) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < size) output[i] = 2.0 * input[i] + 1.0;
}
}  // namespace

std::tuple<at::Tensor, uint64_t, uint64_t> affine_cuda(const at::Tensor& input) {
    TORCH_CHECK(input.is_cuda() && input.scalar_type() == at::kDouble && input.is_contiguous(),
                "affine probe requires a contiguous float64 CUDA tensor");
    TORCH_CHECK(input.numel() > 0, "affine probe requires a nonempty tensor");
    const c10::cuda::CUDAGuard guard(input.device());
    auto output = at::empty_like(input);
    auto stream = c10::cuda::getCurrentCUDAStream(input.get_device()).stream();
    affine_kernel<<<(input.numel() + 255) / 256, 256, 0, stream>>>(
        input.data_ptr<double>(), output.data_ptr<double>(), input.numel());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {output, reinterpret_cast<uint64_t>(stream),
            reinterpret_cast<uint64_t>(input.data_ptr<double>())};
}

std::tuple<at::Tensor, uint64_t> sparse_solve_cuda(
    const at::Tensor& offsets, const at::Tensor& columns,
    const at::Tensor& values, const at::Tensor& rhs) {
    for (const auto& tensor : {offsets, columns, values, rhs}) {
        TORCH_CHECK(tensor.is_cuda() && tensor.is_contiguous() && tensor.dim() == 1,
                    "sparse probe requires contiguous one-dimensional CUDA tensors");
        TORCH_CHECK(tensor.device() == rhs.device(), "sparse probe tensors must share a device");
    }
    TORCH_CHECK(offsets.scalar_type() == at::kInt && columns.scalar_type() == at::kInt,
                "sparse probe CSR indices must be int32");
    TORCH_CHECK(values.scalar_type() == at::kDouble && rhs.scalar_type() == at::kDouble,
                "sparse probe values and rhs must be float64");
    const int64_t n = rhs.numel();
    TORCH_CHECK(n > 0 && offsets.numel() == n + 1 && columns.numel() == values.numel(),
                "inconsistent CSR dimensions");
    const c10::cuda::CUDAGuard guard(rhs.device());
    auto output = at::empty_like(rhs);
    auto stream = c10::cuda::getCurrentCUDAStream(rhs.get_device()).stream();
    SparseObjects state;
    check(cudssCreate(&state.handle), "cudssCreate");
    check(cudssSetStream(state.handle, stream), "cudssSetStream");
    check(cudssConfigCreate(&state.config), "cudssConfigCreate");
    check(cudssDataCreate(state.handle, &state.data), "cudssDataCreate");
    check(cudssMatrixCreateCsr(
        &state.matrix, n, n, values.numel(), offsets.data_ptr<int>(), nullptr,
        columns.data_ptr<int>(), values.data_ptr<double>(),
        CUDA_R_32I, CUDA_R_32I, CUDA_R_64F,
        CUDSS_MTYPE_SPD, CUDSS_MVIEW_LOWER, CUDSS_BASE_ZERO), "cudssMatrixCreateCsr");
    check(cudssMatrixCreateDn(&state.rhs, n, 1, n, rhs.data_ptr<double>(),
        CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR), "cudssMatrixCreateDn(rhs)");
    check(cudssMatrixCreateDn(&state.solution, n, 1, n, output.data_ptr<double>(),
        CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR), "cudssMatrixCreateDn(solution)");
    for (int phase : {CUDSS_PHASE_ANALYSIS, CUDSS_PHASE_FACTORIZATION, CUDSS_PHASE_SOLVE}) {
        check(cudssExecute(state.handle, phase, state.config, state.data,
            state.matrix, state.solution, state.rhs), "cudssExecute");
    }
    // This is a one-shot correctness check; finish work before releasing the factors.
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    return {output, reinterpret_cast<uint64_t>(stream)};
}

std::vector<int> compiled_versions() {
    int runtime = 0, driver = 0, major = 0, minor = 0, patch = 0;
    C10_CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
    C10_CUDA_CHECK(cudaDriverGetVersion(&driver));
    check(cudssGetProperty(MAJOR_VERSION, &major), "cudssGetProperty(major)");
    check(cudssGetProperty(MINOR_VERSION, &minor), "cudssGetProperty(minor)");
    check(cudssGetProperty(PATCH_LEVEL, &patch), "cudssGetProperty(patch)");
    return {CUDART_VERSION, runtime, driver, major, minor, patch};
}
