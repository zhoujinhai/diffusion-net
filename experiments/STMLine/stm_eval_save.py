# Test script: run trained model on validation set and save
# per-vertex results as [x, y, z, pred_label].
#
# Usage:
#   python stm_eval_save.py --input_features xyz_normal_curv --output_dir data/val_predictions
#
import os
import sys
import glob
import argparse
import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "../../src/"))
import diffusion_net
from stm_dataset import STMDataset


parser = argparse.ArgumentParser()
parser.add_argument("--input_features", type=str, default="xyz_normal_curv",
                    help="features: xyz / xyz_normal / xyz_normal_curv / hks")
parser.add_argument("--output_dir", type=str, default="data/val_predictions",
                    help="where to save per-mesh prediction files")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--no-se", action="store_false", dest="se", default=True, help="disable Squeeze-and-Excitation channel attention")
args = parser.parse_args()

device = torch.device(args.device)
input_features = args.input_features
k_eig = 128
n_class = 2
base_path = os.path.dirname(__file__)
op_cache_dir = os.path.join(base_path, "data", "op_cache")
model_path = os.path.join(base_path, "data/saved_models/stm_seg_{}_4x128.pth".format(input_features))
dataset_path = "/home/heygears/Data/Teeth/STMLines"
out_dir = os.path.join(base_path, args.output_dir)
use_se = args.se
print("use_se: ", use_se)

assert os.path.exists(model_path), "model not found: " + model_path

# ---- model (must match training config) ----
C_in = {"xyz": 3, "xyz_normal": 6, "xyz_normal_curv": 8, "hks": 16}[input_features]
model = diffusion_net.layers.DiffusionNet(
    C_in=C_in,
    C_out=n_class,
    C_width=128,
    N_block=4,
    last_activation=lambda x: torch.nn.functional.log_softmax(x, dim=-1),
    outputs_at="vertices",
    dropout=True,
    with_gradient_features=True,
    use_se=use_se
)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device).eval()
print("loaded model from " + model_path)


def build_features(verts, normals, curv, evals, evecs):
    if input_features == "xyz":
        return verts
    elif input_features == "xyz_normal":
        return torch.cat([verts, normals], dim=-1)
    elif input_features == "xyz_normal_curv":
        return torch.cat([verts, normals, curv], dim=-1)
    elif input_features == "hks":
        return diffusion_net.geometry.compute_hks_autoscale(evals, evecs, 16)
    else:
        raise ValueError("unknown input_features: " + input_features)


# ---- load validation dataset (operators are cached from training) ----
test_dataset = STMDataset(dataset_path, train=False, k_eig=k_eig,
                          use_cache=True, op_cache_dir=op_cache_dir)

# file order matches the dataset's internal sorted order
val_dir = os.path.join(dataset_path, "val")
label_files = sorted(glob.glob(os.path.join(val_dir, "*_label.npy")))
assert len(label_files) == len(test_dataset), \
    "mismatch: {} label files vs {} dataset samples".format(len(label_files), len(test_dataset))

os.makedirs(out_dir, exist_ok=True)
print("running inference on {} validation meshes...".format(len(test_dataset)))

for idx in range(len(test_dataset)):
    verts, faces, frames, mass, L, evals, evecs, gradX, gradY, labels, normals, curv = test_dataset[idx]

    verts = verts.to(device); faces = faces.to(device); frames = frames.to(device)
    mass = mass.to(device); L = L.to(device); evals = evals.to(device); evecs = evecs.to(device)
    gradX = gradX.to(device); gradY = gradY.to(device)
    normals = normals.to(device); curv = curv.to(device)

    features = build_features(verts, normals, curv, evals, evecs)
    with torch.no_grad():
        preds = model(features, mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY)
    pred_labels = torch.max(preds, dim=1).indices.cpu().numpy()  # (N,)

    # ground-truth labels and raw (real) coordinates from the original file
    raw = np.load(label_files[idx])           # (N, 9)
    xyz = raw[:, :3]                          # real-world coordinates (N,3)
    gt = raw[:, 8].astype(np.int64)           # (N,)

    base = os.path.basename(label_files[idx])[:-len("_label.npy")]

    # [x, y, z, pred_label]
    out = np.concatenate([xyz, pred_labels[:, None].astype(np.float64)], axis=1)
    out_txt = os.path.join(out_dir, base + "_pred.txt")
    out_npy = os.path.join(out_dir, base + "_pred.npy")
    np.savetxt(out_txt, out, fmt="%.6f %.6f %.6f %d")
    np.save(out_npy, out)

    acc = float((pred_labels == gt).mean())
    print("  [{}] n_vert={} pred_acc={:.4f} -> {}".format(idx, len(gt), acc, os.path.basename(out_txt)))

print("DONE. results saved to " + out_dir)
