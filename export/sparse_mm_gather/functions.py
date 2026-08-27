# functions.py
#
# Python wrapper for the sparse_mm_gather custom operator, following the
# pointops pattern: a torch.autograd.Function whose .forward() calls the
# compiled C++ kernel and whose .symbolic() emits an ONNX custom-op node
# "sparse::SparseMMGather" (so torch.onnx.export produces a single custom node
# instead of slow generic ScatterElements).
#
import os
import torch
from torch.autograd import Function

# path to the compiled extension relative to this file
_DIR = os.path.dirname(os.path.abspath(__file__))
import sys
if _DIR not in sys.path:
    sys.path.append(_DIR)

try:
    import sparse_mm_gather_cpu as _ext
except ImportError:
    _ext = None


def _is_torch_function(func):
    # placeholder kept for parity with pointops style; unused
    return False


class SparseMMGather(Function):
    """out = gradX @ x, gradX given as COO triplets (rows, cols, vals)."""

    @staticmethod
    def forward(ctx, x, rows, cols, vals):
        # x: (V, C); rows/cols/vals: (nnz,)
        if _ext is None:
            raise RuntimeError(
                "sparse_mm_gather_cpu not built. Run: "
                "python setup.py build_ext --inplace in the sparse_mm_gather dir")
        return _ext.sparse_mm_gather_forward(x, rows, cols, vals)

    @staticmethod
    def symbolic(g, x, rows, cols, vals):
        # Emit an ONNX custom op node in domain "sparse".
        return g.op(
            "ai.onnx.contrib::SparseMMGather", x, rows, cols, vals,
            domain_s="ai.onnx.contrib",
        )


# public alias matching the plain python implementation used in export_onnx_sparse
sparse_mm_gather = SparseMMGather.apply


def sparse_mm_gather_ref(x, rows, cols, vals, V):
    """Pure-torch reference (gather + scatter_add) used to validate the C++ op."""
    nnz = cols.shape[0]
    C = x.shape[1]
    xg = x[cols]                                   # (nnz, C)
    contrib = vals[:, None] * xg                   # (nnz, C)
    out = torch.zeros(V, C, dtype=x.dtype, device=x.device)
    out = torch.scatter_add(out, 0, rows[:, None].expand_as(contrib), contrib)
    return out
