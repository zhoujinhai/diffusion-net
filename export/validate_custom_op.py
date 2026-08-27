#!/usr/bin/env python
"""
validate_custom_op.py

Validate the custom ONNX op `ai.onnx.contrib::SparseMMGather` (backed by the
compiled C++ kernel in sparse_mm_gather/) using onnxruntime-extensions PyOp.

This is a FUNCTIONALITY check: PyOp runs a Python callback, so the timing it
reports includes Python-bridge + tensor-copy overhead and is NOT a faithful
measure of the C++ speedup. For real inference speedup use a native C++
OrtCustomOp instead.

Usage:
  conda activate learn_zjh
  # 1) export a model with the custom op node:
  python export/export_onnx_sparse.py --input_features xyz_normal_curv --no-se \
      --use_custom_op --output experiments/STMLine/data/exported/stm_seg_xyz_normal_curv_4x128_custom.onnx
  # 2) validate:
  python export/validate_custom_op.py --input_features xyz_normal_curv --no-se
"""
import os, sys, time
import numpy as np
import torch

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(_PROJ, "export", "sparse_mm_gather"))
sys.path.append(os.path.join(_PROJ, "export"))
sys.path.append(os.path.join(_PROJ, "src"))

import onnxruntime as ort
from onnxruntime_extensions import PyOp, onnx_op, get_library_path
from functions import SparseMMGather
from export_onnx_sparse import load_dataset_ops, build_features


# --- register the custom op backed by the compiled C++ kernel ---
@onnx_op(op_type="SparseMMGather", domain="ai.onnx.contrib",
         inputs=[PyOp.dt_float, PyOp.dt_int64, PyOp.dt_int64, PyOp.dt_float],
         outputs=[PyOp.dt_float], since_version=1)
def _sparse_mm_gather_pyop(x, rows, cols, vals):
    x = torch.from_numpy(x)
    rows = torch.from_numpy(rows).long()
    cols = torch.from_numpy(cols).long()
    vals = torch.from_numpy(vals)
    out = SparseMMGather.apply(x, rows, cols, vals)
    return out.numpy()


def _bench(sess, feeds, n):
    for _ in range(2):
        sess.run(None, feeds)
    t0 = time.perf_counter()
    for _ in range(n):
        sess.run(None, feeds)
    return (time.perf_counter() - t0) / n


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Validate custom SparseMMGather ONNX op")
    parser.add_argument("--input_features", type=str, default="xyz_normal_curv")
    parser.add_argument("--se", action="store_true", default=True)
    parser.add_argument("--no-se", action="store_false", dest="se")
    parser.add_argument("--k_eig", type=int, default=128)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--custom_model", type=str, default=None,
                        help="path to the custom-op ONNX (default: auto)")
    parser.add_argument("--standard_model", type=str, default=None,
                        help="path to the standard ScatterElements ONNX (default: auto)")
    parser.add_argument("--dataset", type=str, default="/home/heygears/Data/Teeth/STMLines")
    args = parser.parse_args()

    exp_path = os.path.join(_PROJ, "experiments", "STMLine")
    export_dir = os.path.join(exp_path, "data", "exported")

    # ---- locate models ----
    name = "stm_seg_{}_4x128".format(args.input_features)
    if args.se:
        name += "_use_se"
    std_path = args.standard_model or os.path.join(export_dir, name + "_sparse.onnx")
    cust_path = args.custom_model or os.path.join(export_dir, name + "_custom.onnx")
    assert os.path.exists(std_path), "standard model not found: " + std_path
    assert os.path.exists(cust_path), (
        "custom model not found: " + cust_path + "  -> export it first with --use_custom_op")

    # ---- load one real mesh ----
    ops = load_dataset_ops(args.dataset, args.k_eig, args.input_features,
                           op_cache_dir=os.path.join(exp_path, "data", "op_cache"))
    torch.manual_seed(0)
    features = build_features(ops["verts"], ops["normals"], ops["curv"],
                              ops["evals"], ops["evecs"], args.input_features)
    feeds = {
        "features": features.numpy(), "mass": ops["mass"].numpy(),
        "evals": ops["evals"].numpy(), "evecs": ops["evecs"].numpy(),
        "gx_rows": ops["gx_rows"].numpy(), "gx_cols": ops["gx_cols"].numpy(), "gx_vals": ops["gx_vals"].numpy(),
        "gy_rows": ops["gy_rows"].numpy(), "gy_cols": ops["gy_cols"].numpy(), "gy_vals": ops["gy_vals"].numpy(),
    }
    print("mesh: V=%d, K=%d, gradX nnz=%d" % (ops["verts"].shape[0], ops["evals"].shape[0], ops["gx_vals"].shape[0]))

    # ---- sessions ----
    so_std = ort.SessionOptions()
    so_std.intra_op_num_threads = args.threads
    sess_std = ort.InferenceSession(std_path, so_std, providers=["CPUExecutionProvider"])

    so_cust = ort.SessionOptions()
    so_cust.intra_op_num_threads = args.threads
    so_cust.register_custom_ops_library(get_library_path())
    sess_cust = ort.InferenceSession(cust_path, so_cust, providers=["CPUExecutionProvider"])

    # ---- correctness ----
    o_std = sess_std.run(None, feeds)[0]
    o_cust = sess_cust.run(None, feeds)[0]
    agree = float((np.argmax(o_std, axis=1) == np.argmax(o_cust, axis=1)).mean())
    print("pred agreement (standard vs custom): {:.6f}".format(agree))
    print("VERIFY", "OK" if agree >= 0.9999 else "FAIL")

    # ---- timing (PyOp includes python bridge overhead) ----
    ms_std = _bench(sess_std, feeds, args.runs)
    ms_cust = _bench(sess_cust, feeds, args.runs)
    print("\n[NOTE] PyOp timing includes Python-bridge + tensor-copy overhead.")
    print("       It is NOT a faithful measure of the C++ speedup;")
    print("       use a native C++ OrtCustomOp for real inference timing.")
    print("standard  (ScatterElements): %.2f s" % ms_std)
    print("custom op (C++ via PyOp)   : %.2f s" % ms_cust)
    print("speedup: %.2fx" % (ms_std / ms_cust))


if __name__ == "__main__":
    main()
