#include <torch/extension.h>

#include <cstdint>
#include <tuple>
#include <vector>

std::tuple<at::Tensor, uint64_t, uint64_t> affine_cuda(const at::Tensor& input);
std::tuple<at::Tensor, uint64_t> sparse_solve_cuda(
    const at::Tensor& offsets, const at::Tensor& columns,
    const at::Tensor& values, const at::Tensor& rhs);
std::vector<int> compiled_versions();

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("affine", &affine_cuda);
    module.def("sparse_solve", &sparse_solve_cuda);
    module.def("compiled_versions", &compiled_versions);
}
