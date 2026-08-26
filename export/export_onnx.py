#!/usr/bin/env python
"""
Export a trained DiffusionNet (STMLine dental segmentation) to ONNX.

Usage:
  conda activate learn_zjh
  python export/export_onnx.py --input_features xyz_normal_curv            # SE model (default)
  python export/export_onnx.py --input_features xyz_normal_curv --no-se    # non-SE model 
python export/export_onnx.py --input_features xyz_normal_curv  --cache_dir experiments/STMLine/data/op_cache  # with cache

The model & input features must match the saved checkpoint (see stm_segmentation.py).

ONNX inputs (all DENSE tensors; DiffusionNet runs in 'spectral' mode where the
sparse Laplacian L is unused, and gradX/gradY are consumed through torch.mm so
they are provided as dense matrices):

  features : (V, C_in)   per-vertex input features
  mass     : (V,)        mass vector
  evals    : (K,)        eigenvalues
  evecs    : (V, K)      eigenvectors
  gradX    : (V, V)      gradient-X operator (dense)
  gradY    : (V, V)      gradient-Y operator (dense)

  probs    : (V, C_out)  per-vertex log-probabilities (log-softmax)

V (num vertices) and K (num eigenvalues) are dynamic axes.

The export sample is drawn from REAL cached operators in op_cache (mass/evals/
evecs/verts), which are numerically well-conditioned and give a faithful graph.
gradX/gradY are kept as synthetic dense matrices, because densifying a real
high-resolution mesh (V x V) would consume huge amounts of RAM. At inference
time you must feed dense gradX/gradY matching the mesh size.


"""
import os, sys, glob, argparse
import numpy as np
import torch

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # diffusion-net/
sys.path.append(os.path.join(_PROJ, "src"))
import diffusion_net
from diffusion_net.layers import DiffusionNet

try:
    import onnxruntime as ort
except ImportError:
    ort = None


class DiffusionNetONNXWrapper(torch.nn.Module):
    """Wrap DiffusionNet so ONNX receives dense, fixed inputs (no sparse L)."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, features, mass, evals, evecs, gradX, gradY):
        return self.net(features, mass, L=None, evals=evals, evecs=evecs,
                        gradX=gradX, gradY=gradY)


def build_features(verts, normals, curv, evals, evecs, input_features):
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


def load_cached_operators(op_cache_dir, k_eig, prefer_smallest=True, v_max=None):
    """Load REAL cached operators from op_cache (a .npz written by get_operators).

    Returns (verts, mass, evals, evecs) as torch tensors from a real mesh. This
    gives a faithful, numerically well-conditioned export sample without needing
    to recompute operators. gradX/gradY are NOT returned (they are dense in the
    ONNX graph; see main). If op_cache_dir is None or empty, returns None.
    """
    if not op_cache_dir or not os.path.isdir(op_cache_dir):
        return None
    files = sorted(glob.glob(os.path.join(op_cache_dir, "*_0.npz")))
    if not files:
        return None

    # pick the smallest-V mesh (or the first if prefer_smallest=False)
    chosen = None
    chosen_V = None
    for f in files:
        try:
            z = np.load(f, allow_pickle=True)
            V = z["verts"].shape[0]
            if chosen is None:
                chosen = (f, z); chosen_V = V
            if prefer_smallest and V < chosen_V and (v_max is None or V <= v_max):
                chosen = (f, z); chosen_V = V
            elif v_max is not None and V <= v_max:
                chosen = (f, z); chosen_V = V
        except Exception:
            continue

    if chosen is None:
        return None
    f, z = chosen
    K_cache = z["evals"].shape[0]
    K = min(k_eig, K_cache)
    verts = torch.from_numpy(z["verts"]).float()
    mass = torch.from_numpy(z["mass"]).float()
    evals = torch.from_numpy(z["evals"][:K]).float()
    evecs = torch.from_numpy(z["evecs"][:, :K]).float()
    print("  loaded real cached operators from {} (V={}, K={})".format(os.path.basename(f), verts.shape[0], K))
    return verts, mass, evals, evecs


def make_sample_operators(V, K, cached=None):
    """Build a numerically-stable sample to drive ONNX export.

    If `cached` (a (verts, mass, evals, evecs) tuple from load_cached_operators)
    is given, use the REAL mass/evals/evecs/verts (real mesh V). Otherwise build
    a synthetic orthonormal sample. gradX/gradY are always synthetic dense
    matrices (real ones cannot be densified for large meshes)."""
    if cached is not None:
        verts, mass, evals, evecs = cached
        V = verts.shape[0]
        torch.manual_seed(0)
        normals = torch.randn(V, 3)
        curv = torch.randn(V, 2)
    else:
        torch.manual_seed(0)
        verts = torch.randn(V, 3)
        normals = torch.randn(V, 3)
        curv = torch.randn(V, 2)
        mass = torch.rand(V) + 0.5
        evals = torch.linspace(0.01, 1.0, K)
        M = torch.randn(V, K)
        evecs, _ = torch.linalg.qr(M)      # orthonormal columns
    # diagonal-ish sparse gradient operators, densified for ONNX input
    row = torch.arange(V)
    gx = torch.sparse_coo_tensor(torch.stack([row, row]), torch.randn(V), (V, V)).to_dense()
    gy = torch.sparse_coo_tensor(torch.stack([row, row]), torch.randn(V), (V, V)).to_dense()
    return verts, normals, curv, mass, evals, evecs, gx, gy


def main():
    parser = argparse.ArgumentParser(description="Export STMLine DiffusionNet to ONNX")
    parser.add_argument("--input_features", type=str, default="xyz_normal_curv",
                        help="features: xyz / xyz_normal / xyz_normal_curv / hks")
    parser.add_argument("--se", action="store_true", default=True,
                        help="use SE channel attention (must match saved model)")
    parser.add_argument("--no-se", action="store_false", dest="se")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--k_eig", type=int, default=128)
    parser.add_argument("--V", type=int, default=1024,
                        help="synthetic sample size (only used when cache is empty)")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="op_cache dir; if set, load REAL mass/evals/evecs from it")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--output", type=str, default=None, help="output .onnx path")
    parser.add_argument("--verify", action="store_true", default=True,
                        help="verify with onnxruntime vs PyTorch")
    parser.add_argument("--no-verify", action="store_false", dest="verify")
    args = parser.parse_args()

    device = torch.device(args.device)
    exp_path = os.path.join(_PROJ, "experiments", "STMLine")

    # ---- pick the saved model (SE version preferred when --se) ----
    model_path = os.path.join(exp_path, "data/saved_models/stm_seg_{}_4x128.pth".format(args.input_features))
    if args.se:
        se_path = os.path.join(exp_path, "data/saved_models/stm_seg_{}_4x128_use_se.pth".format(args.input_features))
        if os.path.exists(se_path):
            model_path = se_path
    assert os.path.exists(model_path), "model not found: " + model_path
    print("loading model:", model_path)

    # ---- construct the model (must match training config) ----
    n_class = 2
    C_in = {"xyz": 3, "xyz_normal": 6, "xyz_normal_curv": 8, "hks": 16}[args.input_features]
    net = DiffusionNet(
        C_in=C_in, C_out=n_class, C_width=128, N_block=4,
        last_activation=lambda x: torch.nn.functional.log_softmax(x, dim=-1),
        outputs_at="vertices", dropout=True, with_gradient_features=True,
        use_se=args.se,
    )
    net.load_state_dict(torch.load(model_path, map_location=device))
    net = net.to(device).eval()
    print("model constructed (C_in={}, C_out={}, use_se={})".format(C_in, n_class, args.se))

    # ---- build the export sample (real cached ops if available, else synthetic) ----
    cached = None
    if args.cache_dir:
        print("loading cached operators from:", args.cache_dir)
        cached = load_cached_operators(args.cache_dir, args.k_eig)
    V = args.V
    K = args.k_eig
    verts, normals, curv, mass, evals, evecs, gradX, gradY = make_sample_operators(V, K, cached=cached)

    verts = verts.to(device); normals = normals.to(device); curv = curv.to(device)
    mass = mass.to(device); evals = evals.to(device); evecs = evecs.to(device)
    gradX = gradX.to(device); gradY = gradY.to(device)

    features = build_features(verts, normals, curv, evals, evecs, args.input_features)
    print("sample: V={}, K={}, C_in={}".format(verts.shape[0], evals.shape[0], C_in))

    wrapper = DiffusionNetONNXWrapper(net).to(device).eval()

    # ---- output path ----
    if args.output is None:
        export_dir = os.path.join(exp_path, "data", "exported")
        os.makedirs(export_dir, exist_ok=True)
        name = "stm_seg_{}_4x128".format(args.input_features)
        if args.se:
            name += "_use_se"
        out_path = os.path.join(export_dir, name + ".onnx")
    else:
        out_path = args.output
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    print("exporting to:", out_path)

    # ---- dynamic axes: V (vertices) and K (eigenvalues) may vary ----
    dynamic_axes = {
        "features": {0: "V"},
        "mass":     {0: "V"},
        "evals":    {0: "K"},
        "evecs":    {0: "V", 1: "K"},
        "gradX":    {0: "V", 1: "V"},
        "gradY":    {0: "V", 1: "V"},
        "probs":    {0: "V"},
    }

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (features, mass, evals, evecs, gradX, gradY),
            out_path,
            input_names=["features", "mass", "evals", "evecs", "gradX", "gradY"],
            output_names=["probs"],
            dynamic_axes=dynamic_axes,
            opset_version=args.opset,
            do_constant_folding=False,
        )
    print("ONNX export done.")

    # ---- optional verification with onnxruntime ----
    if args.verify:
        if ort is None:
            print("onnxruntime not installed; skipping verification")
            return
        print("verifying with onnxruntime...")
        sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
        feeds = {
            "features": features.cpu().numpy(),
            "mass":     mass.cpu().numpy(),
            "evals":    evals.cpu().numpy(),
            "evecs":    evecs.cpu().numpy(),
            "gradX":    gradX.cpu().numpy(),
            "gradY":    gradY.cpu().numpy(),
        }
        ort_out = sess.run(None, feeds)[0]  # (V, C_out)

        with torch.no_grad():
            torch_out = wrapper(features, mass, evals, evecs, gradX, gradY).cpu().numpy()

        max_abs = float(np.abs(ort_out - torch_out).max())
        print("max |ONNX - PyTorch| = {:.3e}".format(max_abs))
        if max_abs < 1e-4:
            print("VERIFY OK: ONNX output matches PyTorch")
        else:
            print("WARNING: outputs differ; check the exported graph")

        pred = np.argmax(ort_out, axis=1)
        print("sample pred-label counts:", dict(zip(*np.unique(pred, return_counts=True))))


if __name__ == "__main__":
    main()
