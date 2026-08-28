# C++ ONNX Runtime 推理 DiffusionNet

在 /home/heygears/zjh/learn/diffusion-net/export/cpp_infer/ 下的 C++ 工程，用 onnxruntime 加载含自定义算子 ai.onnx.contrib::SparseMMGather 的 ONNX 模型推理。

## 加速效果（V=125315, gradX nnz=872775, 16 threads）

| 模型 | 推理耗时 | 预测标签 |
|---|---|---|
| 标准版（内置 ScatterElements） | 7295 ms | 105111 / 20204 |
| 自定义算子版（SparseMMGather） | 1721 ms | 105111 / 20204 |

约 4.2 倍加速，预测结果与 Python 完全一致。

## 文件结构
```
src/
  CMakeLists.txt   # 链接 onnxruntime 1.23.2 库
  main.cpp         # C++ 推理主程序（含 SparseMMGather OrtCustomOp）
  build/           # 编译输出（diffusionnet_infer）
```

配套 Python 脚本（在 export/ 目录）：
```txt
- export_onnx_sparse.py --use_custom_op  导出含自定义算子的 ONNX
- make_input_bin.py  生成推理输入二进制文件
```

## 编译
```bash
conda activate learn_zjh
cd /home/heygears/zjh/learn/diffusion-net/export/cpp_infer
rm -rf build
mkdir build
cd build
cmake ..
make
```

依赖：
```bash
- onnxruntime 头文件：onnxruntime/include
- onnxruntime 库：onnxruntime/libonnxruntime.so
```

## 生成输入文件
```
conda activate learn_zjh
cd /home/heygears/zjh/learn/diffusion-net
python export/make_input_bin.py --input_features xyz_normal_curv --output experiments/STMLine/data/exported/input.bin
```

输入二进制格式（小端）：
```txt
  int64 V, K, NX, NY, C_in
  float[V*C_in]  features
  float[V]       mass
  float[K]       evals
  float[V*K]     evecs
  int64[NX] gx_rows; int64[NX] gx_cols; float[NX] gx_vals
  int64[NY] gy_rows; int64[NY] gy_cols; float[NY] gy_vals
```

## 运行推理
```bash
cd /home/heygears/zjh/learn/diffusion-net/export/cpp_infer/build
LD_LIBRARY_PATH=/home/heygears/anaconda3/envs/learn_zjh/lib/python3.10/site-packages/onnxruntime/capi \
  ./diffusionnet_infer <model.onnx> <input.bin> <threads>
```

例如：
```bash
  ./diffusionnet_infer \
    /home/heygears/zjh/learn/diffusion-net/experiments/STMLine/data/exported/stm_seg_xyz_normal_curv_4x128_custom.onnx \
    /home/heygears/zjh/learn/diffusion-net/experiments/STMLine/data/exported/input.bin  16
```

输出：打印推理耗时、输出 shape、预测标签计数，并写 pred_labels.txt（每顶点一行标签）。

## 说明

- 自定义算子：main.cpp 里 SparseMMGatherOp 实现了 gradX @ x（COO 三元组 -> 排序 + OpenMP 并行按行累加），替换 onnxruntime 慢的通用 ScatterElements，这是加速 4.2x 的关键。
- 输入文件由 make_input_bin.py 从真实网格算子生成（复用 op_cache / dataset）。
## geometry_ops C++ 验证 (2026-08-27)

geometry_ops.h/.cpp 重新实现了 compute_operators()，已验证与 Python get_operators() 一致：
- normalize_positions / vertex_areas / cotan_laplacian / build_tangent_frames / build_grad_ops 全部通过（float 精度级）
- generalized_eigs 修复：Spectra 返回降序特征值，已改为升序（对齐 scipy eigsh）
- gradX/gradY 在 f64 输入下与算法完全一致（1e-13）；与生产 float32 参考的 ~1e-4 相对差为 float32/float64 输入精度差异
