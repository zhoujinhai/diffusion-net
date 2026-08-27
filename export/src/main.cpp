// main.cpp
//
// C++ ONNX Runtime inference for the DiffusionNet dental-segmentation model that
// was exported with the custom op `ai.onnx.contrib::SparseMMGather` (see
// export_onnx_sparse.py --use_custom_op). This file:
//   1. implements SparseMMGather as an onnxruntime OrtCustomOp (fast CPU kernel:
//      sort-by-row + OpenMP parallel, avoiding the slow generic ScatterElements),
//   2. loads the .onnx, registers the custom op, builds the 10 input tensors,
//   3. runs inference and prints per-vertex prediction labels + timing.
//
// Inputs are read from a binary file produced by export/make_input_bin.py
// (see README in this folder). Format (little-endian):
//   int64 V, K, NX, NY, C_in
//   float[V*C_in]  features
//   float[V]       mass
//   float[K]       evals
//   float[V*K]     evecs
//   int64[NX] gx_rows; int64[NX] gx_cols; float[NX] gx_vals
//   int64[NY] gy_rows; int64[NY] gy_cols; float[NY] gy_vals
//
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <chrono>
#include <cstdint>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "onnxruntime_cxx_api.h"

// ---------------------------------------------------------------------------
// SparseMMGather custom op: out = gradX @ x (COO triplets)
// ---------------------------------------------------------------------------
void sparse_mm_gather_cpu(const float* x, const int64_t* rows, const int64_t* cols,
                          const float* vals, int64_t nnz, int64_t V, int64_t C, float* out) {
    std::fill(out, out + V * C, 0.0f);
    if (nnz == 0) return;

    std::vector<int64_t> idx(nnz);
    std::iota(idx.begin(), idx.end(), int64_t(0));
    std::sort(idx.begin(), idx.end(), [&](int64_t a, int64_t b) { return rows[a] < rows[b]; });

    std::vector<int64_t> start(V + 1, 0);
    for (int64_t k = 0; k < nnz; ++k) start[rows[idx[k]] + 1]++;
    std::partial_sum(start.begin(), start.end(), start.begin());

#pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < V; ++r) {
        int64_t lo = start[r], hi = start[r + 1];
        if (lo >= hi) continue;
        float* orow = out + r * C;
        for (int64_t k = lo; k < hi; ++k) {
            int64_t c = cols[idx[k]];
            float v = vals[idx[k]];
            const float* xrow = x + c * C;
            for (int64_t cc = 0; cc < C; ++cc) orow[cc] += v * xrow[cc];
        }
    }
}
// --- OrtCustomOp: kernel ---
struct SparseMMGatherKernel {
    explicit SparseMMGatherKernel(const OrtApi& ort_api, const OrtKernelInfo*) : ort_(ort_api) {}
    void Compute(OrtKernelContext* context) {
        Ort::KernelContext ctx(context);

        // inputs: x(V,C) float, rows(nnz) int64, cols(nnz) int64, vals(nnz) float
        auto x_in = ctx.GetInput(0);
        auto rows_in = ctx.GetInput(1);
        auto cols_in = ctx.GetInput(2);
        auto vals_in = ctx.GetInput(3);

        const float* x = x_in.GetTensorData<float>();
        const int64_t* rows = rows_in.GetTensorData<int64_t>();
        const int64_t* cols = cols_in.GetTensorData<int64_t>();
        const float* vals = vals_in.GetTensorData<float>();

        auto x_shape = x_in.GetTensorTypeAndShapeInfo().GetShape();
        int64_t V = x_shape[0];
        int64_t C = x_shape[1];
        int64_t nnz = rows_in.GetTensorTypeAndShapeInfo().GetShape()[0];

        std::vector<int64_t> out_shape = {V, C};
        auto out = ctx.GetOutput(0, out_shape);
        float* out_data = out.GetTensorMutableData<float>();

        sparse_mm_gather_cpu(x, rows, cols, vals, nnz, V, C, out_data);
    }
private:
    const OrtApi& ort_;
};

// --- OrtCustomOp: op description ---
struct SparseMMGatherOp : Ort::CustomOpBase<SparseMMGatherOp, SparseMMGatherKernel> {
    explicit SparseMMGatherOp(const char* provider) : provider_(provider) {}

    void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const {
        return new SparseMMGatherKernel(api, info);
    }
    const char* GetName() const { return "SparseMMGather"; }
    const char* GetExecutionProviderType() const { return provider_; }

    size_t GetInputTypeCount() const { return 4; }
    ONNXTensorElementDataType GetInputType(size_t index) const {
        if (index == 0) return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;   // x
        if (index == 1 or index == 2) return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64; // rows, cols
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;                    // vals
    }
    OrtCustomOpInputOutputCharacteristic GetInputCharacteristic(size_t) const {
        return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
    }

    size_t GetOutputTypeCount() const { return 1; }
    ONNXTensorElementDataType GetOutputType(size_t) const {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
    OrtCustomOpInputOutputCharacteristic GetOutputCharacteristic(size_t) const {
        return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
    }
private:
    const char* provider_;
};
// --- binary input reader (see header comment for format) ---
bool read_int64(std::ifstream& f, int64_t& v) { return f.read(reinterpret_cast<char*>(&v), 8).good(); }
template <typename T>
bool read_block(std::ifstream& f, std::vector<T>& buf, size_t n) {
    buf.resize(n);
    return n == 0 or f.read(reinterpret_cast<char*>(buf.data()), n * sizeof(T)).good();
}

// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    // args: <model.onnx> <input.bin> [threads]
    std::string model_path = argc > 1 ? argv[1] : "stm_seg_xyz_normal_curv_4x128_custom.onnx";
    std::string input_path = argc > 2 ? argv[2] : "input.bin";
    int threads = argc > 3 ? std::atoi(argv[3]) : 16;

    // ---- read input file ----
    std::ifstream f(input_path, std::ios::binary);
    if (!f.is_open()) { std::cerr << "cannot open input file: " << input_path << std::endl; return -1; }
    int64_t V=0, K=0, NX=0, NY=0, C_in=0;
    read_int64(f, V); read_int64(f, K); read_int64(f, NX); read_int64(f, NY); read_int64(f, C_in);
    std::cout << "V=" << V << " K=" << K << " NX=" << NX << " NY=" << NY << " C_in=" << C_in << std::endl;

    std::vector<float> features; read_block(f, features, V * C_in);
    std::vector<float> mass;     read_block(f, mass, V);
    std::vector<float> evals;    read_block(f, evals, K);
    std::vector<float> evecs;    read_block(f, evecs, V * K);
    std::vector<int64_t> gx_rows, gx_cols; std::vector<float> gx_vals;
    read_block(f, gx_rows, NX); read_block(f, gx_cols, NX); read_block(f, gx_vals, NX);
    std::vector<int64_t> gy_rows, gy_cols; std::vector<float> gy_vals;
    read_block(f, gy_rows, NY); read_block(f, gy_cols, NY); read_block(f, gy_vals, NY);
    f.close();

    // ---- register custom op domain ----
    Ort::CustomOpDomain domain("ai.onnx.contrib");
    SparseMMGatherOp smg_op("CPUExecutionProvider");
    domain.Add(&smg_op);

    Ort::SessionOptions so;
    so.Add(domain);
    so.SetIntraOpNumThreads(threads);
    so.SetInterOpNumThreads(1);

    // ---- load model ----
    Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "diffusionnet-infer");
    Ort::Session session(env, model_path.c_str(), so);


    // ---- build input tensors ----
    Ort::MemoryInfo mem_info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
    std::vector<int64_t> f_shape{V, C_in}, m_shape{V}, e_shape{K}, ev_shape{V, K};
    std::vector<int64_t> rx_shape{NX}, rc_shape{NX}, rv_shape{NX};
    std::vector<int64_t> yx_shape{NY}, yc_shape{NY}, yv_shape{NY};

    std::vector<Ort::Value> inputs;
    inputs.push_back(Ort::Value::CreateTensor<float>(mem_info, features.data(), features.size(), f_shape.data(), f_shape.size()));
    inputs.push_back(Ort::Value::CreateTensor<float>(mem_info, mass.data(), mass.size(), m_shape.data(), m_shape.size()));
    inputs.push_back(Ort::Value::CreateTensor<float>(mem_info, evals.data(), evals.size(), e_shape.data(), e_shape.size()));
    inputs.push_back(Ort::Value::CreateTensor<float>(mem_info, evecs.data(), evecs.size(), ev_shape.data(), ev_shape.size()));
    inputs.push_back(Ort::Value::CreateTensor<int64_t>(mem_info, gx_rows.data(), gx_rows.size(), rx_shape.data(), rx_shape.size()));
    inputs.push_back(Ort::Value::CreateTensor<int64_t>(mem_info, gx_cols.data(), gx_cols.size(), rc_shape.data(), rc_shape.size()));
    inputs.push_back(Ort::Value::CreateTensor<float>(mem_info, gx_vals.data(), gx_vals.size(), rv_shape.data(), rv_shape.size()));
    inputs.push_back(Ort::Value::CreateTensor<int64_t>(mem_info, gy_rows.data(), gy_rows.size(), yx_shape.data(), yx_shape.size()));
    inputs.push_back(Ort::Value::CreateTensor<int64_t>(mem_info, gy_cols.data(), gy_cols.size(), yc_shape.data(), yc_shape.size()));
    inputs.push_back(Ort::Value::CreateTensor<float>(mem_info, gy_vals.data(), gy_vals.size(), yv_shape.data(), yv_shape.size()));

    std::vector<const char*> input_names = {"features", "mass", "evals", "evecs",
                                            "gx_rows", "gx_cols", "gx_vals",
                                            "gy_rows", "gy_cols", "gy_vals"};
    const char* output_name = "probs";


    // ---- run inference (timed) ----
    // warmup
    try { session.Run(Ort::RunOptions{nullptr}, input_names.data(), inputs.data(), inputs.size(), &output_name, 1); }
    catch (const Ort::Exception& e) { std::cerr << "warmup failed: " << e.what() << std::endl; }

    const int n_runs = 10;
    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<Ort::Value> out;
    for (int i = 0; i < n_runs; ++i) {
        out = session.Run(Ort::RunOptions{nullptr}, input_names.data(), inputs.data(), inputs.size(), &output_name, 1);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / n_runs;
    std::cout << "avg inference time (" << threads << " threads): " << ms << " ms" << std::endl;

    // ---- parse output ----
    Ort::Value& ov = out[0];
    auto info = ov.GetTensorTypeAndShapeInfo();
    auto shape = info.GetShape();
    size_t elems = info.GetElementCount();
    const float* probs = ov.GetTensorData<float>();
    std::cout << "output shape: [" << (shape.empty() ? -1 : shape[0]);
    if (shape.size() > 1) std::cout << ", " << shape[1];
    std::cout << "]  elements=" << elems << std::endl;

    int64_t C_out = shape.size() > 1 ? shape[1] : 1;
    std::vector<int> pred(V);
    std::vector<int> counts(C_out, 0);
    for (int64_t i = 0; i < V; ++i) {
        int best = 0; float bv = probs[i * C_out];
        for (int c = 1; c < C_out; ++c)
            if (probs[i * C_out + c] > bv) { bv = probs[i * C_out + c]; best = c; }
        pred[i] = best;
        counts[best]++;
    }
    std::cout << "pred label counts:";
    for (int c = 0; c < C_out; ++c) std::cout << " " << c << ":" << counts[c];
    std::cout << std::endl;

    // write labels to a txt (one per line)
    std::ofstream of("pred_labels.txt");
    for (int64_t i = 0; i < V; ++i) of << pred[i] << "\n";
    of.close();
    std::cout << "wrote pred_labels.txt" << std::endl;

    return 0;
}
