# C++ ONNX Runtime 推理 DiffusionNet

在 /home/heygears/zjh/learn/diffusion-net/export/cpp_infer/ 下的 C++ 工程，用 onnxruntime 加载含自定义算子 ai.onnx.contrib::SparseMMGather 的 ONNX 模型推理。

## 加速效果（V=125315, gradX nnz=872775, 16 threads）

| 模型 | 推理耗时 | 预测标签 |
|---|---|---|
| 标准版（内置 ScatterElements） | 7295 ms | 105111 / 20204 |
| 自定义算子版（SparseMMGather） | 1721 ms | 105111 / 20204 |

约 4.2 倍加速，预测结果与 Python 完全一致。

## 文件结构

cpp_infer/
  CMakeLists.txt   # 链接 onnxruntime 1.23.2 库
  main.cpp         # C++ 推理主程序（含 SparseMMGather OrtCustomOp）
  build/           # 编译输出（diffusionnet_infer）

配套 Python 脚本（在 export/ 目录）：
- export_onnx_sparse.py --use_custom_op  导出含自定义算子的 ONNX
- make_input_bin.py  生成推理输入二进制文件

## 编译

conda activate learn_zjh
cd /home/heygears/zjh/learn/diffusion-net/export/cpp_infer
rm -rf build; mkdir build; cd build
cmake ..; make

依赖：
- onnxruntime 头文件：onnxruntime/include
- onnxruntime 库：onnxruntime/libonnxruntime.so

## 生成输入文件

conda activate learn_zjh
cd /home/heygears/zjh/learn/diffusion-net
python export/make_input_bin.py --input_features xyz_normal_curv --output experiments/STMLine/data/exported/input.bin

输入二进制格式（小端）：
  int64 V, K, NX, NY, C_in
  float[V*C_in]  features
  float[V]       mass
  float[K]       evals
  float[V*K]     evecs
  int64[NX] gx_rows; int64[NX] gx_cols; float[NX] gx_vals
  int64[NY] gy_rows; int64[NY] gy_cols; float[NY] gy_vals

## 运行推理

cd /home/heygears/zjh/learn/diffusion-net/export/cpp_infer/build
LD_LIBRARY_PATH=/home/heygears/anaconda3/envs/learn_zjh/lib/python3.10/site-packages/onnxruntime/capi \
  ./diffusionnet_infer <model.onnx> <input.bin> <threads>

例如：
  ./diffusionnet_infer \
    /home/heygears/zjh/learn/diffusion-net/experiments/STMLine/data/exported/stm_seg_xyz_normal_curv_4x128_custom.onnx \
    /home/heygears/zjh/learn/diffusion-net/experiments/STMLine/data/exported/input.bin \
    16

输出：打印推理耗时、输出 shape、预测标签计数，并写 pred_labels.txt（每顶点一行标签）。

## 说明

- 自定义算子：main.cpp 里
