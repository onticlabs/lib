// pybind11 bindings for the 3D RoPE CUDA kernel (defined in kernels.cu).
//
// Mirrors LitePT's split: kernel in .cu, exports in .cpp. The compiled
// extension is loaded as `point_rope_cuda._C` and exposes a single function:
//   pointrope(tokens, positions, base, fwd) → None  (mutates tokens in place)

#include <torch/extension.h>

// Forward declaration of the kernel launcher defined in kernels.cu
void pointrope_cuda(torch::Tensor tokens, const torch::Tensor pos, const float base, const float fwd);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("pointrope", &pointrope_cuda,
          "3D RoPE (continuous coord, in-place on tokens)");
}
