# setup.py
#
# Build the sparse_mm_gather custom operator (CPU + OpenMP) as a PyTorch
# extension. The compiled .so can be imported in Python and used as the backend
# of a torch.autograd.Function whose .symbolic() emits an ONNX custom-op node.
#
# Usage:
#   conda activate learn_zjh
#   cd /home/heygears/zjh/learn/diffusion-net/export/sparse_mm_gather
#   python setup.py build_ext --inplace
#   # produces: sparse_mm_gather_cpu.<pyver>.so  (or sparse_mm_gather.so)
#
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension
import os

# force OpenMP if available (gcc/clang). Use -fopenmp on Linux.
extra_cxx = ["-O3", "-fopenmp", "-std=c++17"]
extra_link = ["-fopenmp"]

ext = CppExtension(
    name="sparse_mm_gather_cpu",
    sources=["src/sparse_mm_gather.cpp"],
    extra_compile_args={"cxx": extra_cxx},
    extra_link_args=extra_link,
)

setup(
    name="sparse_mm_gather",
    version="1.0.0",
    description="Custom CPU sparse matrix-multiply operator for DiffusionNet ONNX export",
    ext_modules=[ext],
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
)
