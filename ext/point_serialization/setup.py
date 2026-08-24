"""Build: from this dir run `pip install -e .` (or `python setup.py build_ext --inplace`).

Vendored from ChristianSchott/point_serialization_cuda — CUDA Morton (Z-order)
and Hilbert encode kernels, used as a fast path for PTV3-style point
serialization. Layout mirrors `fwomo_3d/libs/point_rope/`.
"""
import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

here = os.path.dirname(os.path.abspath(__file__))
sources = [
    os.path.join(here, f)
    for f in ("bindings.cpp", "hilbert_cuda.cu", "hilbert_cuda_approx.cu", "morton_cuda.cu")
]

# Architectures we actually run on: A40 (sm_86), A100 (sm_80), H100 (sm_90),
# B200 (sm_100). Override with TORCH_CUDA_ARCH_LIST if needed.
default_archs = [
    "-gencode", "arch=compute_80,code=sm_80",
    "-gencode", "arch=compute_86,code=sm_86",
    "-gencode", "arch=compute_90,code=sm_90",
]
nvcc_flags = ["-O3"]
if not os.environ.get("TORCH_CUDA_ARCH_LIST"):
    nvcc_flags += default_archs

setup(
    name="serialize_cuda",
    version="0.1",
    install_requires=["torch"],
    # install the Python wrapper package too — without this only the raw
    # serialize_cuda .so lands in site-packages (uint64 returns, no validation)
    packages=["point_serialization"],
    package_dir={"point_serialization": "."},
    ext_modules=[
        CUDAExtension(
            name="serialize_cuda",
            sources=sources,
            extra_compile_args={"cxx": ["-O3"], "nvcc": nvcc_flags},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
