#!/usr/bin/env python
"""
dump_mesh_bin.py

Dump a single mesh's verts/faces to a binary file for the C++ geometry_ops test.

Format (little-endian):
  int64 V, F
  float64[V*3] verts
  int32[F*3]   faces
"""
import os, sys, struct
import numpy as np

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(_PROJ, "experiments", "STMLine"))
sys.path.append(os.path.join(_PROJ, "src"))

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, default=0, help="mesh index in val set")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    from stm_dataset import STMDataset
    OC = os.path.join(_PROJ, "experiments", "STMLine", "data", "op_cache")
    ds = STMDataset("/home/heygears/Data/Teeth/STMLines", train=False, k_eig=128, op_cache_dir=OC)
    verts = ds[args.idx][0].cpu().numpy().astype(np.float64)  # (V,3)
    faces = ds[args.idx][1].cpu().numpy().astype(np.int32)    # (F,3)
    V, F = verts.shape[0], faces.shape[0]
    out = args.output or os.path.join(_PROJ, "export", "cpp_infer", "test_mesh.bin")
    with open(out, "wb") as f:
        f.write(struct.pack("<qq", V, F))
        f.write(verts.tobytes())
        f.write(faces.tobytes())
    print("wrote %s V=%d F=%d" % (out, V, F))

if __name__ == "__main__":
    main()
