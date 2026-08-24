// 3D RoPE CUDA kernel — adapted from LitePT (prs-eth/LitePT) for *float*
// positions. Accepts either continuous metric xyz or LitePT-style integer
// grid_coord (cast to float in the Python wrapper before launch). The kernel
// does `pos * inv_freq` in float regardless of input convention, so the two
// modes are bit-for-bit identical at matched numeric values — pick the one
// you want at the call site:
//   - `point.coord`              → continuous (allows coord-grad in PyTorch path)
//   - `point.grid_coord.float()` → integer-equivalent (LitePT default)
//
// Computes:
//   freq = pos[axis] * inv_freq[i]      (with i ∈ [0, head_dim/6))
//   token pair (u, v) → (u*cos - v*sin, v*cos + u*sin)
// where head_dim is split into 3 axis chunks (size head_dim/3 each); each
// chunk's first half is u (paired with the second half v) under standard
// rotate_half. The kernel mutates tokens in place.
//
// Backward: re-apply the kernel with F0 → -F0 to invert the rotation
// (rotation is orthogonal). This means gradients w.r.t. tokens flow through,
// but gradients w.r.t. coord do *not* (positions are treated as constants).
// Use the pure-PyTorch path in `pointrope_torch.Point3DRoPE` if you need
// coord gradients.

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(t)                                                          \
    {                                                                          \
        TORCH_CHECK((t).is_cuda(), #t " is not on cuda");                      \
        TORCH_CHECK((t).is_contiguous(), #t " is not contiguous");             \
    }


template <typename scalar_t>
__global__ void pointrope_cuda_kernel(
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> tokens,
    const float* __restrict__ pos,  // (B, N, 3) continuous xyz, float
    const float base,
    const float fwd)
{
    // tokens: (B, N, H, D)
    const int N = tokens.size(1);
    const int H = tokens.size(2);
    const int D = tokens.size(3);

    extern __shared__ float shared[];
    float* shared_inv_freq = shared + D;

    const int b = blockIdx.x / N;
    const int n = blockIdx.x % N;

    const int Q = D / 6;  // chunk_dim/2; head_dim split into 6 (u_x v_x u_y v_y u_z v_z)

    if (threadIdx.x < Q)
        shared_inv_freq[threadIdx.x] = fwd / powf(base, threadIdx.x / float(Q));
    __syncthreads();

    // which axis (0=x, 1=y, 2=z) does this thread belong to
    const int X = threadIdx.x * 3 / D;
    // index within axis chunk: maps to u-side, paired with v at m+Q
    const int m = (X * D / 3) + (threadIdx.x % Q);

    // continuous coord lookup; multiplication implicitly casts to float
    const float freq = pos[blockIdx.x * 3 + X] * shared_inv_freq[threadIdx.x % Q];
    const float cos_v = cosf(freq);
    const float sin_v = sinf(freq);

    for (int h = 0; h < H; h++) {
        shared[threadIdx.x] = tokens[b][n][h][threadIdx.x];
        __syncthreads();

        const float u = shared[m];
        const float v = shared[m + Q];

        if ((threadIdx.x % (D / 3)) < Q)
            tokens[b][n][h][threadIdx.x] = u * cos_v - v * sin_v;
        else
            tokens[b][n][h][threadIdx.x] = v * cos_v + u * sin_v;
    }
}


void pointrope_cuda(torch::Tensor tokens, const torch::Tensor pos, const float base, const float fwd)
{
    const int B = tokens.size(0);
    const int N = tokens.size(1);
    const int D = tokens.size(3);

    CHECK_CUDA(tokens);
    CHECK_CUDA(pos);
    TORCH_CHECK(tokens.stride(3) == 1 && tokens.stride(2) == D, "tokens not contiguous");
    TORCH_CHECK(pos.size(0) == B && pos.size(1) == N && pos.size(2) == 3, "bad pos.shape");
    TORCH_CHECK(D % 6 == 0, "token head_dim must be multiple of 6");
    TORCH_CHECK(pos.scalar_type() == at::kFloat, "pos must be float32");

    const int THREADS_PER_BLOCK = D;
    const int N_BLOCKS = B * N;
    const int SHARED_MEM = sizeof(float) * (D + D / 6);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, tokens.scalar_type(), "pointrope_cuda", ([&] {
        pointrope_cuda_kernel<scalar_t><<<N_BLOCKS, THREADS_PER_BLOCK, SHARED_MEM>>>(
            tokens.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            pos.data_ptr<float>(),
            base, fwd);
    }));
}
