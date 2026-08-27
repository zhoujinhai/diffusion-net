#!/usr/bin/env python
"""
make_input_bin.py

Generate the binary input file consumed by cpp_infer/main.cpp for a single mesh.

Format (little-endian, see main.cpp header):
  int64 V, K, NX, NY, C_in
  float[V*C_in]  features
  float[V]       mass
  float[K]       evals
  float[V*K]     evecs
  int64[NX] gx_rows; int64[NX] gx_cols; float[NX] gx_vals
  int64[NY] gy_rows; int64[NY] gy_cols; float[NY] gy_vals

Usage:
  conda activate learn_zjh
  python export/make_input_bin.py --input_features xyz_normal_curv --no-se \
      --output experiments/STMLine/data/exported/input.bin
"""
import os, sys, struct
import numpy as np

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(_PROJ, "export"))
sys.path.append(os.path.join(_PROJ, "src"))
from export_onnx_sparse import load_dataset_ops, build_features


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_features", type=str, default="xyz_normal_curv")
    parser.add_argument("--k_eig", type=int, default=128)
    parser.add_argument("--dataset", type=str, default="/home/heygears/Data/Teeth/STMLines")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    exp_path = os.path.join(_PROJ, "experiments", "STMLine")
    ops = load_dataset_ops(args.dataset, args.k_eig, args.input_features,
                           op_cache_dir=os.path.join(exp_path, "data", "op_cache"))

    import torch
    torch.manual_seed(0)
    features = build_features(ops["verts"], ops["normals"], ops["curv"],
                              ops["evals"], ops["evecs"], args.input_features)

    V = ops["verts"].shape[0]
    K = ops["evals"].shape[0]
    NX = ops["gx_vals"].shape[0]
    NY = ops["gy_vals"].shape[0]
    C_in = features.shape[1]
    print("V=%d K=%d NX=%d NY=%d C_in=%d" % (V, K, NX, NY, C_in))

    def f32(t): return np.ascontiguousarray(t.cpu().numpy(), dtype=np.float32)
    def i64(t): return np.ascontiguousarray(t.cpu().numpy(), dtype=np.int64)

    out_path = args.output or os.path.join(exp_path, "data", "exported", "input.bin")
    with open(out_path, "wb") as f:
        for x in (V, K, NX, NY, C_in):
            f.write(struct.pack("<q", x))
        f.write(f32(features).tobytes())
        f.write(f32(ops["mass"]).tobytes())
        f.write(f32(ops["evals"]).tobytes())
        f.write(f32(ops["evecs"]).tobytes())
        f.write(i64(ops["gx_rows"]).tobytes()); f.write(i64(ops["gx_cols"]).tobytes()); f.write(f32(ops["gx_vals"]).tobytes())
        f.write(i64(ops["gy_rows"]).tobytes()); f.write(i64(ops["gy_cols"]).tobytes()); f.write(f32(ops["gy_vals"]).tobytes())
    print("wrote", out_path, "size=%.1f MB" % (os.path.getsize(out_path) / 1e6))


if __name__ == "__main__":
    main()
