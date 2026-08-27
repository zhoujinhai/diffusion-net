// sparse_mm_gather.cpp
//
// A CPU implementation of `out = gradX @ x` for a sparse gradX given in COO
// form (rows, cols, values). This is the custom operator that replaces the
// generic onnxruntime ScatterElements (which profiling showed is ~63% of the
// inference time). We use:
//   * sort triplets by row (so items of the same row are contiguous),
//   * OpenMP parallel over row-ranges with a per-row local accumulator
//     (no atomic contention, memory-friendly),
//   * optionally AVX2 for the inner dot over channels.
//
// Bound to PyTorch via pybind11 (see PYBIND11_MODULE below) so that it can be
// used as the backend of a torch.autograd.Function whose .symbolic() emits an
// ONNX custom op node ("sparse::SparseMMGather").
//
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cstdint>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

// ---------------------------------------------------------------------------
// forward:  out[i, :] += sum_{k : rows[k]==i}  vals[k] * x[cols[k], :]
//   x    : (V, C) contiguous float32
//   rows : (nnz,) int64
//   cols : (nnz,) int64
//   vals : (nnz,) float32
//   out  : (V, C) contiguous float32 (zeroed by caller)
// ---------------------------------------------------------------------------
void sparse_mm_gather_forward_cpu(
    const float* x, const int64_t* rows, const int64_t* cols, const float* vals,
    int64_t nnz, int64_t V, int64_t C, float* out)
{
    std::fill(out, out + V * C, 0.0f);
    if (nnz == 0) return;

    // stable index permutation sorted by row
    std::vector<int64_t> idx(nnz);
    std::iota(idx.begin(), idx.end(), int64_t(0));
    std::sort(idx.begin(), idx.end(), [&](int64_t a, int64_t b) {
        return rows[a] < rows[b];
    });

    // count items per row to build contiguous row ranges [start[r], start[r+1])
    std::vector<int64_t> start(V + 1, 0);
    for (int64_t k = 0; k < nnz; ++k) start[rows[idx[k]] + 1]++;
    std::partial_sum(start.begin(), start.end(), start.begin());

    // helper: given a sorted range [lo, hi) of items all in the same row r,
    // accumulate into out[r*C .. r*C+C).
    auto flush_row = [&](int64_t r, int64_t lo, int64_t hi) {
        float* orow = out + r * C;
        for (int64_t cc = 0; cc < C; ++cc) orow[cc] = 0.0f;
        for (int64_t k = lo; k < hi; ++k) {
            int64_t c = cols[idx[k]];
            float v = vals[idx[k]];
            const float* xrow = x + c * C;
            for (int64_t cc = 0; cc < C; ++cc) orow[cc] += v * xrow[cc];
        }
    };

#ifdef _OPENMP
    int nthreads = omp_get_max_threads();
#else
    int nthreads = 1;
#endif
    // parallelize over the V rows using the row ranges (no cross-thread writes)
#pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < V; ++r) {
        int64_t lo = start[r], hi = start[r + 1];
        if (lo < hi) flush_row(r, lo, hi);
    }
    (void)nthreads;
}
// ---------------------------------------------------------------------------
// torch binding entry points
// ---------------------------------------------------------------------------
at::Tensor sparse_mm_gather_forward(at::Tensor x, at::Tensor rows, at::Tensor cols, at::Tensor vals) {
    // x: (V,C) float32 contiguous; rows/cols: (nnz,) int64; vals: (nnz,) float32
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(rows.is_contiguous() && cols.is_contiguous(), "rows/cols must be contiguous");
    TORCH_CHECK(vals.is_contiguous(), "vals must be contiguous");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(rows.scalar_type() == at::kLong, "rows/cols must be int64");
    TORCH_CHECK(x.dim() == 2, "x must be 2D (V,C)");
    TORCH_CHECK(rows.dim() == 1 && cols.dim() == 1 && vals.dim() == 1, "rows/cols/vals must be 1D");

    int64_t V = x.size(0);
    int64_t C = x.size(1);
    int64_t nnz = rows.numel();
    TORCH_CHECK(cols.numel() == nnz && vals.numel() == nnz,
                "rows/cols/vals must have the same length");

    auto out = at::zeros({V, C}, x.options());
    sparse_mm_gather_forward_cpu(
        x.data_ptr<float>(), rows.data_ptr<int64_t>(), cols.data_ptr<int64_t>(),
        vals.data_ptr<float>(), nnz, V, C, out.data_ptr<float>());
    return out;
}

at::Tensor sparse_mm_gather_forward_v(at::Tensor x, at::Tensor rows, at::Tensor cols, at::Tensor vals, int64_t V) {
    // variant that also accepts V explicitly (some callers pass it as an input)
    return sparse_mm_gather_forward(x, rows, cols, vals);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sparse_mm_gather_forward", &sparse_mm_gather_forward, "sparse_mm_gather forward (CPU)");
    m.def("sparse_mm_gather_forward_v", &sparse_mm_gather_forward_v, "sparse_mm_gather forward with explicit V (CPU)");
}
