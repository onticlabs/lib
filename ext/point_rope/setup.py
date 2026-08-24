"""Build: cd to this dir and run `pip install -e .` (or `python setup.py install`).

Layout mirrors LitePT (prs-eth/LitePT) `libs/pointrope/`:
  kernels.cu       — CUDA kernel
  pointrope.cpp    — pybind11 bindings (exports `pointrope` op)
"""
import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

here = os.path.dirname(os.path.abspath(__file__))
sources = [os.path.join(here, f) for f in ("kernels.cu", "pointrope.cpp")]

setup(
    name="point_rope_cuda",
    version="0.1",
    install_requires=["torch"],
    # without this, a non-editable install ships only the compiled _C.so and
    # drops point_rope_cuda/__init__.py, leaving a broken half-package
    packages=["point_rope_cuda"],
    ext_modules=[
        CUDAExtension(
            name="point_rope_cuda._C",
            sources=sources,
            extra_compile_args={"cxx": ["-O2"], "nvcc": ["-O2"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
