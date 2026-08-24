#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ inline uint64_t hilbert3d_approx_single(uint32_t x, uint32_t y, uint32_t z, int num_bits) {
    uint32_t X = x;
    uint32_t Y = y;
    uint32_t Z = z;

    // --- Skilling transform ---
    uint32_t Q = 1u << (num_bits - 1);
    while (Q > 1) {
        uint32_t P = Q - 1;

        if (X & Q) { X ^= P; Y ^= P; Z ^= P; }
        if (Y & Q) { X ^= P; Y ^= P; Z ^= P; }
        if (Z & Q) { X ^= P; Y ^= P; Z ^= P; }

        Q >>= 1;
    }

    // --- Gray encode ---
    X ^= (X >> 1);
    Y ^= (Y >> 1);
    Z ^= (Z >> 1);

    // --- Interleave bits ---
    uint64_t h = 0;
    for (int i = num_bits - 1; i >= 0; --i) {
        h <<= 3;
        h |= ((uint64_t)((X >> i) & 1) << 2) |
             ((uint64_t)((Y >> i) & 1) << 1) |
             ((uint64_t)((Z >> i) & 1));
    }

    return h;
}

__global__ void hilbert_approx_kernel(
    const uint32_t* __restrict__ coords,
    uint64_t* __restrict__ out,
    int N,
    int num_bits
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    uint32_t x = coords[3 * idx + 0];
    uint32_t y = coords[3 * idx + 1];
    uint32_t z = coords[3 * idx + 2];

    out[idx] = hilbert3d_approx_single(x, y, z, num_bits);
}

torch::Tensor hilbert_encode_approx_cuda(torch::Tensor coords, int num_bits) {
    TORCH_CHECK(coords.is_cuda(), "coords must be CUDA");
    TORCH_CHECK(coords.size(1) == 3, "coords must be (N, 3)");

    int N = coords.size(0);

    auto coords_u32 = coords.to(torch::kUInt32).contiguous();
    auto out = torch::empty({N}, torch::dtype(torch::kUInt64).device(coords.device()));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    hilbert_approx_kernel<<<blocks, threads>>>(
        coords_u32.data_ptr<uint32_t>(),
        out.data_ptr<uint64_t>(),
        N,
        num_bits
    );

    return out;
}