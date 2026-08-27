#!/usr/bin/env python
"""
Profile a DiffusionNet ONNX model with onnxruntime to find per-op time.

Usage:
  conda activate learn_zjh
  python export/profile_onnx.py --input_features xyz_normal_curv --no-se
  python export/profile_onnx.py --model <path> --cache_dir <op_cache>

Generates an onnxruntime profiling .json file, then aggregates per-op durations
and prints a sorted table (total time, count, avg) so you can see which op
dominates (e.g. ScatterElements/Gather from the sparse gradient multiply).
"""
import os, sys, glob, json, argparse
import numpy as np
import onnxruntime as ort

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(_PROJ, "src"))
sys.path.append(os.path.join(_PROJ, "experiments", "STMLine"))

try:
    from export.export_onnx_sparse import load_dataset_ops, build_features
except Exception:
    from export_onnx_sparse import load_dataset_ops, build_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_features", type=str, default="xyz_normal_curv")
    parser.add_argument("--se", action="store_true", default=True)
    parser.add_argument("--no-se", action="store_false", dest="se")
    parser.add_argument("--model", type=str, default=None,
                        help="ONNX model path (default: the sparse exported model)")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="op_cache dir to load real operators")
    parser.add_argument("--dataset", type=str, default="/home/heygears/Data/Teeth/STMLines")
    parser.add_argument("--k_eig", type=int, default=128)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--runs", type=int, default=5,
                        help="profiled inference runs")
    args = parser.parse_args()

    exp_path = os.path.join(_PROJ, "experiments", "STMLine")

    # ---- pick model ----
    if args.model is None:
        name = "stm_seg_{}_4x128".format(args.input_features)
        if args.se:
            name += "_use_se"
        name += "_sparse.onnx"
        model_path = os.path.join(exp_path, "data", "exported", name)
    else:
        model_path = args.model
    assert os.path.exists(model_path), "model not found: " + model_path
    print("model:", model_path)

    # ---- load one real mesh ----
    ops = load_dataset_ops(args.dataset, args.k_eig, args.input_features,
                           op_cache_dir=args.cache_dir if args.cache_dir else os.path.join(exp_path, "data", "op_cache"))
    torch_manual_seed = __import__("torch").manual_seed
    torch_manual_seed(0)
    features = build_features(ops["verts"], ops["normals"], ops["curv"],
                              ops["evals"], ops["evecs"], args.input_features)
    feeds = {
        "features": features.numpy(), "mass": ops["mass"].numpy(),
        "evals": ops["evals"].numpy(), "evecs": ops["evecs"].numpy(),
        "gx_rows": ops["gx_rows"].numpy(), "gx_cols": ops["gx_cols"].numpy(), "gx_vals": ops["gx_vals"].numpy(),
        "gy_rows": ops["gy_rows"].numpy(), "gy_cols": ops["gy_cols"].numpy(), "gy_vals": ops["gy_vals"].numpy(),
    }
    print("mesh: V=%d, K=%d, gradX nnz=%d" % (ops["verts"].shape[0], ops["evals"].shape[0], ops["gx_vals"].shape[0]))

    # ---- session with profiling ----
    so = ort.SessionOptions()
    so.intra_op_num_threads = args.threads
    so.inter_op_num_threads = 1
    so.enable_profiling = True
    sess = ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])

    for _ in range(2):
        sess.run(None, feeds)   # warmup
    for _ in range(args.runs):
        sess.run(None, feeds)   # profiled runs
    prof_file = sess.end_profiling()
    print("profiling file:", prof_file)

    # ---- parse profiling json & aggregate per-op time ----
    # onnxruntime profiling file is a jsonl: one JSON object per line
    op_time = {}      # op name -> total time (us)
    op_count = {}     # op name -> count
    total = 0.0
    raw = open(prof_file).read().strip()
    # The file is a JSON array where each element is on its own line with a
    # trailing comma. Strip the outer [ ] and split on newlines.
    if raw.startswith("["):
        raw = raw[1:]
    if raw.endswith("]"):
        raw = raw[:-1]
    for line in raw.split("\n"):
        line = line.strip().rstrip(",").strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if rec.get("cat") == "Node" and "dur" in rec:
            # use op_name from args for aggregation across nodes
            name = rec.get("args", {}).get("op_name") or rec.get("name", "")
            dur = float(rec["dur"])   # microseconds
            op_time[name] = op_time.get(name, 0.0) + dur
            op_count[name] = op_count.get(name, 0) + 1
            total += dur

    print("\n===== per-op aggregated time (total=%.1f ms) =====" % (total / 1000.0))
    rows = sorted(op_time.items(), key=lambda kv: -kv[1])
    for name, t in rows:
        pct = 100.0 * t / total if total > 0 else 0.0
        print("  %-22s %9.1f ms  %6.2f%%  (n=%d, avg=%.3f ms)" %
              (name, t / 1000.0, pct, op_count[name], t / op_count[name] / 1000.0))

    # highlight scatter/gather
    print("\n--- sparse-multiply related ops ---")
    sparse_total = 0.0
    for name, t in op_time.items():
        if "Scatter" in name or "Gather" in name or "Mul" in name:
            sparse_total += t
            print("  %-22s %9.1f ms  %6.2f%%" % (name, t / 1000.0, 100.0 * t / total))
    if total > 0:
        print("  SUBTOTAL (Scatter+Gather+Mul): %.1f ms = %.2f%%" % (sparse_total / 1000.0, 100.0 * sparse_total / total))


if __name__ == "__main__":
    main()
