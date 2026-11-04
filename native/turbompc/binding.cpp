#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <climits>
#include <cstdio>
#include <mutex>
#include <vector>

#include "admm_cudss.cuh"
#include "cudss_sparse_kkt.cuh"

namespace py = pybind11;
using at::Tensor;

uint64_t CreateAdmmCudssWorkspace();
int EnsureAdmmCudssWorkspaceCapacity(turbompc::FfiDeviceAllocator&, uint64_t,
                                    int32_t, int32_t, int32_t, int32_t,
                                    int32_t, int32_t, int32_t);
int DestroyAdmmCudssWorkspace(turbompc::FfiDeviceAllocator&, uint64_t);
const char* AdmmCudssLastError();
uint64_t CreateCudssSparseKktWorkspace(const int32_t*, const int32_t*,
                                      int32_t, int32_t, int32_t);
int DestroyCudssSparseKktWorkspace(turbompc::FfiDeviceAllocator&, uint64_t);
const char* CudssSparseKktLastError();

static void check_tensor(const Tensor& value, const std::vector<int64_t>& shape,
                         int device, const char* name) {
  TORCH_CHECK(value.is_cuda() && value.get_device() == device,
              name, " must be on the workspace CUDA device");
  TORCH_CHECK(value.scalar_type() == at::kDouble,
              name, " must use float64");
  TORCH_CHECK(value.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(value.sizes().vec() == shape, name, " has an invalid shape");
}

static ADMMCudssConfig parse_config(const py::dict& p) {
  ADMMCudssConfig c{};
#define GET_INT(name) c.name = py::cast<int32_t>(p[#name])
#define GET_DOUBLE(name) c.name = py::cast<double>(p[#name])
  GET_INT(max_iter);
  GET_INT(check_termination_every);
  GET_INT(check_infeasibility_every);
  GET_INT(adapt_rho_every);
  GET_DOUBLE(eps_abs);
  GET_DOUBLE(eps_rel);
  GET_DOUBLE(eps_prim_inf);
  GET_DOUBLE(eps_dual_inf);
  GET_DOUBLE(sigma);
  GET_DOUBLE(rho_f_factor);
  GET_DOUBLE(alpha);
  GET_DOUBLE(adaptive_rho_tolerance);
  GET_DOUBLE(rho_update_max_factor);
  GET_DOUBLE(rho_min);
  GET_DOUBLE(rho_max);
  GET_DOUBLE(log_barrier_kappa);
#undef GET_INT
#undef GET_DOUBLE
  const int handling = py::cast<int>(p["constraint_handling"]);
  TORCH_CHECK(handling == 0 || handling == 1, "Unknown constraint handling");
  c.constraint_handling = static_cast<ConstraintHandling>(handling);
  c.use_slack = py::cast<bool>(p["use_slack"]);
  TORCH_CHECK(c.max_iter > 0 && c.check_termination_every > 0 &&
              c.check_infeasibility_every > 0 && c.adapt_rho_every >= 0,
              "Invalid ADMM iteration settings");
  TORCH_CHECK(c.sigma > 0 && c.rho_f_factor > 0 && c.alpha > 0 && c.alpha < 2,
              "Invalid ADMM regularization or relaxation");
  return c;
}

class AdmmWorkspace {
 public:
  explicit AdmmWorkspace(int device) : device_(device) {
    c10::cuda::CUDAGuard guard(device_);
    handle_ = CreateAdmmCudssWorkspace();
    TORCH_CHECK(handle_, AdmmCudssLastError());
  }
  AdmmWorkspace(const AdmmWorkspace&) = delete;
  AdmmWorkspace& operator=(const AdmmWorkspace&) = delete;
  ~AdmmWorkspace() {
    try { close(); }
    catch (const std::exception& e) {
      std::fprintf(stderr, "TurboMPC ADMM workspace release: %s\n", e.what());
    }
  }

  void close() {
    std::lock_guard<std::mutex> lock(lifecycle_mutex_);
    if (!handle_) return;
    c10::cuda::CUDAGuard guard(device_);
    turbompc::FfiDeviceAllocator allocator{
        c10::cuda::getCurrentCUDAStream(device_).stream()};
    const auto handle = handle_;
    handle_ = 0;
    TORCH_CHECK(DestroyAdmmCudssWorkspace(allocator, handle) == 0,
                AdmmCudssLastError());
  }

  std::vector<Tensor> solve(const std::vector<Tensor>& a, const py::dict& params) {
    TORCH_CHECK(a.size() == 20, "Expected 12 QP and 8 warm-start tensors");
    TORCH_CHECK(a[1].dim() == 4 && a[4].dim() == 3 && a[5].dim() == 4 && a[7].dim() == 4,
                "Invalid QP tensor ranks");
    const int64_t b = a[1].size(0), t = a[1].size(1), n = a[1].size(2);
    const int64_t nx = a[5].size(2), n0 = a[4].size(1), m = a[7].size(2);
    TORCH_CHECK(b > 0 && t > 1 && n > 0 && nx > 0 && n0 > 0 && m >= 0,
                "Invalid QP dimensions");
    TORCH_CHECK(b * t * n <= INT_MAX && b * t * m <= INT_MAX,
                "QP dimensions exceed native int32 indexing");
    const std::vector<std::vector<int64_t>> shapes = {
      {b,t,n,3*n}, {b,t,n,n}, {b,t-1,n,n}, {b,t,n}, {b,n0,n},
      {b,t-1,nx,n}, {b,t-1,nx,n}, {b,t,m,n}, {b,t,m}, {b,t,m},
      {b,n0}, {b,t-1,nx}, {b,t,n}, {b,t,m}, {b,t,m}, {b,n0},
      {b,t-1,nx}, {b,t,m}, {b}, {b}
    };
    for (size_t i = 0; i < a.size(); ++i)
      check_tensor(a[i], shapes[i], device_, "ADMM input");
    auto cfg = parse_config(params);
    c10::cuda::CUDAGuard guard(device_);
    auto stream = c10::cuda::getCurrentCUDAStream(device_).stream();
    turbompc::FfiDeviceAllocator allocator{stream};
    const auto options = a[1].options().requires_grad(false);
    std::vector<Tensor> out = {
      at::empty({b}, options.dtype(at::kInt)), at::empty({b,t,n}, options),
      at::empty({b,t,m}, options), at::empty({b,t,m}, options),
      at::empty({b,n0}, options), at::empty({b,t-1,nx}, options),
      at::empty({b,t,m}, options), at::empty({b}, options),
      at::empty({b}, options), at::empty({b}, options.dtype(at::kInt)),
      at::empty({b,4}, options)
    };
    {
      // Unlock before reacquiring the GIL, including when a native call throws.
      py::gil_scoped_release release;
      std::lock_guard<std::mutex> lock(lifecycle_mutex_);
      TORCH_CHECK(handle_, "ADMM workspace is closed");
      TORCH_CHECK(EnsureAdmmCudssWorkspaceCapacity(allocator, handle_, b, t, n,
                                                  nx, n0, m, 8) == 0,
                  AdmmCudssLastError());
      LaunchADMMCudssF64(
        allocator, handle_, stream,
        reinterpret_cast<uint32_t*>(out[0].data_ptr<int32_t>()),
        out[1].data_ptr<double>(), out[2].data_ptr<double>(), out[3].data_ptr<double>(),
        out[4].data_ptr<double>(), out[5].data_ptr<double>(), out[6].data_ptr<double>(),
        out[7].data_ptr<double>(), out[8].data_ptr<double>(),
        reinterpret_cast<uint32_t*>(out[9].data_ptr<int32_t>()), out[10].data_ptr<double>(),
        a[0].data_ptr<double>(), a[1].data_ptr<double>(), a[2].data_ptr<double>(),
        a[3].data_ptr<double>(), a[4].data_ptr<double>(), a[5].data_ptr<double>(),
        a[6].data_ptr<double>(), a[7].data_ptr<double>(), a[8].data_ptr<double>(),
        a[9].data_ptr<double>(), a[10].data_ptr<double>(), a[11].data_ptr<double>(),
        a[12].data_ptr<double>(), a[13].data_ptr<double>(), a[14].data_ptr<double>(),
        a[15].data_ptr<double>(), a[16].data_ptr<double>(), a[17].data_ptr<double>(),
        a[18].data_ptr<double>(), a[19].data_ptr<double>(),
        t, n, nx, n0, m, b, cfg);
    }
    return out;
  }

 private:
  int device_;
  std::mutex lifecycle_mutex_;
  uint64_t handle_ = 0;
};

class KktWorkspace {
 public:
  KktWorkspace(const Tensor& row_ptr, const Tensor& col_idx, int device)
      : device_(device) {
    TORCH_CHECK(!row_ptr.is_cuda() && !col_idx.is_cuda() &&
                row_ptr.scalar_type() == at::kInt && col_idx.scalar_type() == at::kInt &&
                row_ptr.is_contiguous() && col_idx.is_contiguous() &&
                row_ptr.dim() == 1 && col_idx.dim() == 1,
                "KKT pattern must use contiguous CPU int32 vectors");
    n_ = row_ptr.numel() - 1;
    nnz_ = col_idx.numel();
    TORCH_CHECK(n_ > 0 && nnz_ > 0 && n_ <= INT_MAX && nnz_ <= INT_MAX,
                "Invalid KKT pattern size");
    const auto* r = row_ptr.data_ptr<int32_t>();
    const auto* c = col_idx.data_ptr<int32_t>();
    TORCH_CHECK(r[0] == 0 && r[n_] == nnz_, "Invalid KKT CSR row pointers");
    for (int64_t i = 0; i < n_; ++i)
      TORCH_CHECK(r[i] <= r[i+1] && r[i] >= 0, "Invalid KKT CSR row ordering");
    for (int64_t i = 0; i < nnz_; ++i)
      TORCH_CHECK(c[i] >= 0 && c[i] < n_, "Invalid KKT CSR column index");
    handle_ = CreateCudssSparseKktWorkspace(r, c, n_, nnz_, 8);
    TORCH_CHECK(handle_, CudssSparseKktLastError());
  }
  KktWorkspace(const KktWorkspace&) = delete;
  KktWorkspace& operator=(const KktWorkspace&) = delete;
  ~KktWorkspace() {
    try { close(); }
    catch (const std::exception& e) {
      std::fprintf(stderr, "TurboMPC KKT workspace release: %s\n", e.what());
    }
  }
  void close() {
    std::lock_guard<std::mutex> lock(lifecycle_mutex_);
    if (!handle_) return;
    c10::cuda::CUDAGuard guard(device_);
    turbompc::FfiDeviceAllocator allocator{
        c10::cuda::getCurrentCUDAStream(device_).stream()};
    const auto handle = handle_;
    handle_ = 0;
    TORCH_CHECK(DestroyCudssSparseKktWorkspace(allocator, handle) == 0,
                CudssSparseKktLastError());
  }
  Tensor solve(const Tensor& values, const Tensor& rhs) {
    check_tensor(values, {nnz_}, device_, "KKT values");
    check_tensor(rhs, {n_}, device_, "KKT right-hand-side");
    c10::cuda::CUDAGuard guard(device_);
    const auto stream = c10::cuda::getCurrentCUDAStream(device_).stream();
    turbompc::FfiDeviceAllocator allocator{stream};
    auto output = at::empty_like(rhs, rhs.options().requires_grad(false));
    {
      py::gil_scoped_release release;
      std::lock_guard<std::mutex> lock(lifecycle_mutex_);
      TORCH_CHECK(handle_, "KKT workspace is closed");
      turbompc::cudss_sparse_kkt::solve_sparse_kkt_f64(
        allocator, handle_, stream, values.data_ptr<double>(), rhs.data_ptr<double>(),
        output.data_ptr<double>(), n_, nnz_);
    }
    return output;
  }

 private:
  int device_;
  std::mutex lifecycle_mutex_;
  int64_t n_, nnz_;
  uint64_t handle_ = 0;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<AdmmWorkspace>(m, "AdmmWorkspace")
      .def(py::init<int>()).def("solve", &AdmmWorkspace::solve)
      .def("close", &AdmmWorkspace::close, py::call_guard<py::gil_scoped_release>());
  py::class_<KktWorkspace>(m, "KktWorkspace")
      .def(py::init<const Tensor&, const Tensor&, int>())
      .def("solve", &KktWorkspace::solve)
      .def("close", &KktWorkspace::close, py::call_guard<py::gil_scoped_release>());
}
