#!/usr/bin/env python
"""
Usage:
  conda activate learn_zjh
  python export/export_onnx.py --input_features xyz_normal_curv            # SE model (default)
  python export/export_onnx.py --input_features xyz_normal_curv --no-se    # non-SE model
  
  python export/export_onnx.py --input_features xyz_normal_curv --cache_dir experiments/STMLine/data/op_cache  # with cache data


Export DiffusionNet to ONNX using a CUSTOM sparse-multiplication operator.

DiffusionNet's gradX/gradY are SPARSE (V,V) matrices used as gradX @ x. In
PyTorch this is cheap (O(nnz)), but ONNX runtime cannot take sparse tensors
and densifying (V x V) blows up memory for large meshes.

This exporter rewrites gradX @ x and gradY @ x into an equivalent custom
operator built from ONNX-native ops (Gather + ScatterElements, reduction=add),
using the COO triplets (rows, cols, values) as DENSE inputs. Memory stays
O(nnz), so high-resolution meshes are supported.

ONNX inputs:
  features : (V, C_in)
  mass     : (V,)
  evals    : (K,)
  evecs    : (V, K)
  gx_rows, gx_cols, gx_vals : (nnzX,)  COO of gradX
  gy_rows, gy_cols, gy_vals : (nnzY,)  COO of gradY
  probs    : (V, C_out) output

Single sample (no batch dim); V, K and nnz are dynamic axes.
"""
import os, sys, argparse
import numpy as np
import torch

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(_PROJ, "src"))
from diffusion_net.layers import DiffusionNet

try:
    import onnxruntime as ort
except ImportError:
    ort = None


def sparse_mm_gather(x, rows, cols, vals, V):
    """gradX @ x using Gather + ScatterElements (no dense VxV).
    x: (V, C); rows/cols/vals: (nnz,) -> out: (V, C)."""
    xg = x[cols]                       # (nnz, C)
    contrib = vals[:, None] * xg       # (nnz, C)
    out = torch.zeros(V, x.shape[1], device=x.device, dtype=x.dtype)
    out = torch.scatter_add(out, 0, rows[:, None].expand_as(contrib), contrib)
    return out


class SparseONNXWrapper(torch.nn.Module):
    """Replicate DiffusionNet.forward but replace gradX/gradY sparse mm with the
    custom gather-scatter operator, using COO triplets as inputs."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def _block(self, block, x, mass, evals, evecs,
               gx_rows, gx_cols, gx_vals, gy_rows, gy_cols, gy_vals):
        # x, mass, evals, evecs have batch dim (B=1)
        x_diffuse = block.diffusion(x, None, mass, evals, evecs)   # spectral: L unused

        # gradient features via custom sparse mm (B=1)
        xd = x_diffuse[0]                                  # (V, Cw)
        V = xd.shape[0]
        x_gradX = sparse_mm_gather(xd, gx_rows, gx_cols, gx_vals, V)
        x_gradY = sparse_mm_gather(xd, gy_rows, gy_cols, gy_vals, V)
        x_grad = torch.stack((x_gradX, x_gradY), dim=-1).unsqueeze(0)   # (1, V, Cw, 2)
        x_grad_features = block.gradient_features(x_grad)               # (1, V, Cw)

        feature_combined = torch.cat((x, x_diffuse, x_grad_features), dim=-1)
        x0_out = block.mlp(feature_combined)
        if block.use_se:
            x0_out = block.se(x0_out)
        x0_out = x0_out + x
        return x0_out

    def forward(self, features, mass, evals, evecs,
                gx_rows, gx_cols, gx_vals, gy_rows, gy_cols, gy_vals):
        # single sample (no batch): add batch dims internally
        x = self.net.first_lin(features.unsqueeze(0))         # (1, V, Cw)
        mass_b = mass.unsqueeze(0)
        evals_b = evals.unsqueeze(0)
        evecs_b = evecs.unsqueeze(0)
        for block in self.net.blocks:
            x = self._block(block, x, mass_b, evals_b, evecs_b,
                            gx_rows, gx_cols, gx_vals, gy_rows, gy_cols, gy_vals)
        x = self.net.last_lin(x)                              # (1, V, C_out)
        x = x.squeeze(0)                                      # (V, C_out)
        return x


def build_features(verts, normals, curv, evals, evecs, input_features):
    if input_features == "xyz":
        return verts
    elif input_features == "xyz_normal":
        return torch.cat([verts, normals], dim=-1)
    elif input_features == "xyz_normal_curv":
        return torch.cat([verts, normals, curv], dim=-1)
    elif input_features == "hks":
        import diffusion_net
        return diffusion_net.geometry.compute_hks_autoscale(evals, evecs, 16)
    raise ValueError("unknown input_features: " + input_features)


def load_real_ops(op_cache_dir, k_eig):
    """Load a real mesh's operators from op_cache (a get_operators .npz). Returns
    dict with verts/mass/evals/evecs and COO triplets of gradX/gradY."""
    import glob
    files = sorted(glob.glob(os.path.join(op_cache_dir, "*_0.npz")))
    if not files:
        return None
    best = None; best_V = None
    for f in files:
        try:
            z = np.load(f, allow_pickle=True)
            V = z["verts"].shape[0]
            if best is None or V < best_V:
                best = z; best_V = V
        except Exception:
            continue
    if best is None:
        return None
    z = best
    K = min(k_eig, int(z["evals"].shape[0]))
    verts = torch.from_numpy(z["verts"]).float()
    mass = torch.from_numpy(z["mass"]).float()
    evals = torch.from_numpy(z["evals"][:K]).float()
    evecs = torch.from_numpy(z["evecs"][:, :K]).float()

    def coo(prefix):
        data = z[prefix + "_data"]
        indices = z[prefix + "_indices"]
        indptr = z[prefix + "_indptr"]
        shape = z[prefix + "_shape"]
        import scipy.sparse as sp
        m = sp.csc_matrix((data, indices, indptr), shape=shape)
        m = m.tocoo()
        return (torch.from_numpy(m.row).long(),
                torch.from_numpy(m.col).long(),
                torch.from_numpy(m.data).float())
    gx_rows, gx_cols, gx_vals = coo("gradX")
    gy_rows, gy_cols, gy_vals = coo("gradY")
    return dict(verts=verts, mass=mass, evals=evals, evecs=evecs,
                gx_rows=gx_rows, gx_cols=gx_cols, gx_vals=gx_vals,
                gy_rows=gy_rows, gy_cols=gy_cols, gy_vals=gy_vals)


def load_dataset_ops(dataset_path, k_eig, input_features, op_cache_dir=None):
    """Load a real mesh's operators from the validation dataset (STMDataset).

    This gives real verts/normals/curv (from the *_label.npy files) as well as the
    sparse gradX/gradY extracted to COO triplets. Returns the same dict shape as
    load_real_ops, plus normals/curv. If loading fails, returns None."""
    try:
        import diffusion_net
    except Exception:
        pass
    # make sure STMDataset is importable
    stm_dir = os.path.join(_PROJ, "experiments", "STMLine")
    if stm_dir not in sys.path:
        sys.path.append(stm_dir)
    from stm_dataset import STMDataset

    ds = STMDataset(dataset_path, train=False, k_eig=k_eig,
                    use_cache=True, op_cache_dir=op_cache_dir)
    if len(ds) == 0:
        return None
    verts, faces, frames, mass, L, evals, evecs, gradX, gradY, labels, normals, curv = ds[0]

    # gradX/gradY are sparse (V,V) tensors -> extract COO triplets
    gx = gradX.coalesce()
    gy = gradY.coalesce()
    return dict(verts=verts, mass=mass, evals=evals, evecs=evecs,
                normals=normals, curv=curv,
                gx_rows=gx.indices()[0], gx_cols=gx.indices()[1], gx_vals=gx.values(),
                gy_rows=gy.indices()[0], gy_cols=gy.indices()[1], gy_vals=gy.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_features", type=str, default="xyz_normal_curv")
    parser.add_argument("--se", action="store_true", default=True)
    parser.add_argument("--no-se", action="store_false", dest="se")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--k_eig", type=int, default=128)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="op_cache dir to load real operators (recommended)")
    parser.add_argument("--dataset", type=str, default="/home/heygears/Data/Teeth/STMLines",
                        help="dataset root (used when --cache_dir is not given)")
    parser.add_argument("--verify", action="store_true", default=True)
    parser.add_argument("--no-verify", action="store_false", dest="verify")
    args = parser.parse_args()

    device = torch.device(args.device)
    exp_path = os.path.join(_PROJ, "experiments", "STMLine")

    # ---- model ----
    model_path = os.path.join(exp_path, "data/saved_models/stm_seg_{}_4x128.pth".format(args.input_features))
    if args.se:
        se_path = os.path.join(exp_path, "data/saved_models/stm_seg_{}_4x128_use_se.pth".format(args.input_features))
        if os.path.exists(se_path):
            model_path = se_path
    assert os.path.exists(model_path), "model not found: " + model_path
    print("loading model:", model_path)

    n_class = 2
    C_in = {"xyz": 3, "xyz_normal": 6, "xyz_normal_curv": 8, "hks": 16}[args.input_features]
    net = DiffusionNet(C_in=C_in, C_out=n_class, C_width=128, N_block=4,
                       last_activation=lambda x: torch.nn.functional.log_softmax(x, dim=-1),
                       outputs_at="vertices", dropout=True, with_gradient_features=True,
                       use_se=args.se)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net = net.to(device).eval()
    print("model constructed (C_in={}, C_out={}, use_se={})".format(C_in, n_class, args.se))

    # ---- sample: real ops from op_cache (if given) or from the dataset ----
    dataset_path = args.dataset
    ops = None
    if args.cache_dir:
        print("loading real operators from op_cache:", args.cache_dir)
        ops = load_real_ops(args.cache_dir, args.k_eig)
    if ops is None:
        print("loading real operators from validation dataset:", dataset_path)
        ops = load_dataset_ops(dataset_path, args.k_eig, args.input_features,
                               op_cache_dir=os.path.join(exp_path, "data", "op_cache"))
    if ops is None:
        raise RuntimeError("could not load any mesh operators from --cache_dir or the dataset")

    verts = ops["verts"].to(device); mass = ops["mass"].to(device)
    evals = ops["evals"].to(device); evecs = ops["evecs"].to(device)
    gx_rows = ops["gx_rows"].to(device); gx_cols = ops["gx_cols"].to(device); gx_vals = ops["gx_vals"].to(device)
    gy_rows = ops["gy_rows"].to(device); gy_cols = ops["gy_cols"].to(device); gy_vals = ops["gy_vals"].to(device)

    # normals/curv come from the dataset if available, otherwise synthetic
    if "normals" in ops and ops["normals"] is not None:
        normals = ops["normals"].to(device)
        curv = ops["curv"].to(device)
    else:
        torch.manual_seed(0)
        V = verts.shape[0]
        normals = torch.randn(V, 3, device=device)
        curv = torch.randn(V, 2, device=device)
    features = build_features(verts, normals, curv, evals, evecs, args.input_features)
    K = evals.shape[0]
    print("sample: V={}, K={}, C_in={}, gradX nnz={}, gradY nnz={}".format(
        verts.shape[0], K, C_in, gx_vals.shape[0], gy_vals.shape[0]))

    wrapper = SparseONNXWrapper(net).to(device).eval()

    # ---- export ----
    if args.output is None:
        export_dir = os.path.join(exp_path, "data", "exported")
        os.makedirs(export_dir, exist_ok=True)
        name = "stm_seg_{}_4x128".format(args.input_features)
        if args.se:
            name += "_use_se"
        name += "_sparse"
        out_path = os.path.join(export_dir, name + ".onnx")
    else:
        out_path = args.output
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    print("exporting to:", out_path)

    inputs = (features, mass, evals, evecs,
              gx_rows, gx_cols, gx_vals, gy_rows, gy_cols, gy_vals)
    in_names = ["features", "mass", "evals", "evecs",
                "gx_rows", "gx_cols", "gx_vals", "gy_rows", "gy_cols", "gy_vals"]
    dyn = {"features": {0: "V"}, "mass": {0: "V"}, "evals": {0: "K"},
           "evecs": {0: "V", 1: "K"},
           "gx_rows": {0: "NX"}, "gx_cols": {0: "NX"}, "gx_vals": {0: "NX"},
           "gy_rows": {0: "NY"}, "gy_cols": {0: "NY"}, "gy_vals": {0: "NY"},
           "probs": {0: "V"}}
    with torch.no_grad():
        torch.onnx.export(wrapper, inputs, out_path,
                          input_names=in_names, output_names=["probs"],
                          dynamic_axes=dyn, opset_version=args.opset,
                          do_constant_folding=False)
    print("ONNX export done.")

    # ---- verify ----
    if args.verify:
        if ort is None:
            print("onnxruntime not installed; skip verify")
            return
        print("verifying with onnxruntime...")
        sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
        feeds = {"features": features.cpu().numpy(), "mass": mass.cpu().numpy(),
                 "evals": evals.cpu().numpy(), "evecs": evecs.cpu().numpy(),
                 "gx_rows": gx_rows.cpu().numpy(), "gx_cols": gx_cols.cpu().numpy(), "gx_vals": gx_vals.cpu().numpy(),
                 "gy_rows": gy_rows.cpu().numpy(), "gy_cols": gy_cols.cpu().numpy(), "gy_vals": gy_vals.cpu().numpy()}
        ort_out = sess.run(None, feeds)[0]
        with torch.no_grad():
            torch_out = wrapper(features, mass, evals, evecs,
                                gx_rows, gx_cols, gx_vals, gy_rows, gy_cols, gy_vals).cpu().numpy()
        max_abs = float(np.abs(ort_out - torch_out).max())
        print("max |ONNX - PyTorch| = {:.3e}".format(max_abs))
        ort_pred = np.argmax(ort_out, axis=1)
        torch_pred = np.argmax(torch_out, axis=1)
        agree = float((ort_pred == torch_pred).mean())
        print("pred-label agreement (ONNX vs PyTorch): {:.6f}".format(agree))
        print("pred-label counts:", dict(zip(*np.unique(ort_pred, return_counts=True))))
        if agree >= 0.9999:
            print("VERIFY OK (labels match)")
        else:
            print("VERIFY FAIL (label mismatch)")


if __name__ == "__main__":
    main()
