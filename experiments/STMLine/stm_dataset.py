import shutil
import os
import sys
import glob
import zlib
import threading
from collections import OrderedDict
import numpy as np

import torch
from torch.utils.data import Dataset

sys.path.append(os.path.join(os.path.dirname(__file__), "../../src/"))  # add the path to the DiffusionNet src
import diffusion_net
from diffusion_net.utils import random_rotation_matrix, toNP, hash_arrays
from diffusion_net.geometry import get_operators


def _stable_seed(s):
    # Deterministic integer seed from a string (stable across processes)
    return zlib.crc32(s.encode('utf-8')) & 0xffffffff


def _copy_ops_cached_on_disk(base, base_verts, base_normals, faces_t, n_aug, op_cache_dir, k_eig, seed=0):
    """Count how many of a mesh's n_aug augmented copies ALREADY have their geometric
    operators cached on disk under op_cache_dir. Rotations are deterministic (seeded by
    base + "_copy{idx}"), so each copy's rotated+normalized verts are reproducible, which
    makes the disk-cache hash reproducible too. Returns number of copies (0..n_aug) cached."""
    if op_cache_dir is None:
        return 0
    cached = 0
    for copy_idx in range(n_aug):
        if copy_idx == 0:
            R = np.eye(3)
        else:
            rs = np.random.RandomState(seed=(seed + _stable_seed(base + "_copy{}".format(copy_idx))) % (2**31))
            R = random_rotation_matrix(randgen=rs)
        R_t = torch.from_numpy(R).float()
        v = torch.matmul(base_verts, R_t)
        n = torch.matmul(base_normals, R_t)
        v = diffusion_net.geometry.normalize_positions(v)
        hk = str(hash_arrays((toNP(v), toNP(faces_t))))
        path = os.path.join(op_cache_dir, hk + "_0.npz")
        if os.path.exists(path):
            try:
                z = np.load(path, allow_pickle=True)
                if z["k_eig"].item() >= k_eig and "L_data" in z:
                    cached += 1
            except Exception:
                pass
    return cached


class STMDataset(Dataset):
    """
    STM (straightening line) segmentation dataset.

    Data layout under root_dir:
        train/       : per-vertex features, each file <name>_label.npy with shape (N, 9)
                       columns = [x, y, z, nx, ny, nz, min_curv, max_curv, label]
        train_face/  : face indices, each file <name>_face.npy with shape (F, 3) rows [v0, v1, v2]
        val/         : validation per-vertex features (same format as train/)
        val_face/    : validation face indices

    Labels are binary (0 / 1).

    Data augmentation: when n_aug > 1, each mesh is duplicated into n_aug copies.
    Copy 0 keeps the original orientation; copies 1..n_aug-1 are rotated by a fixed
    (deterministic) random rotation. Rotation is applied to the COORDINATES and the
    NORMALS together, BEFORE computing the geometric operators (mass, L, evecs,
    gradX, gradY, ...). Each rotated copy therefore gets its own set of cached
    operators that are consistent with its rotated vertices, so this is valid
    augmentation (one sample -> n_aug samples).

    Memory: geometric operators (mass/L/evecs/gradX/gradY/frames) are computed
    LAZILY (on demand) inside __getitem__ instead of all being precomputed and held
    in RAM at init time. A small LRU in-memory cache keeps only the most recently
    used operators, and a disk cache (op_cache_dir) is reused to avoid recomputing.
    This keeps peak RAM low even for many large meshes / large n_aug.
    """

    def __init__(self, root_dir, train, k_eig, use_cache=True, op_cache_dir=None, n_aug=1, seed=0,
                 max_op_cache=8):
        self.train = train  # bool
        self.root_dir = root_dir
        self.k_eig = k_eig
        self.cache_dir = os.path.join(root_dir, "cache")
        self.op_cache_dir = op_cache_dir
        self.n_class = 2
        self.n_aug = max(1, int(n_aug))  # copies per mesh
        self.max_op_cache = max(1, int(max_op_cache))  # max operator sets held in RAM

        if self.train:
            vert_dir = os.path.join(root_dir, "train")
            face_dir = os.path.join(root_dir, "train_face")
        else:
            vert_dir = os.path.join(root_dir, "val")
            face_dir = os.path.join(root_dir, "val_face")

        # Collect all label files (base names shared with face files)
        label_files = sorted(glob.glob(os.path.join(vert_dir, "*_label.npy")))
        print("loading {} files from {} (n_aug={})".format(len(label_files), vert_dir, self.n_aug))

        self.verts_list = []
        self.faces_list = []
        self.labels_list = []      # per-vertex int labels
        self.normals_list = []     # per-vertex normals (N, 3)
        self.curv_list = []        # per-vertex curvature [min_curv, max_curv] (N, 2)
        self.rot_list = []         # (3,3) rotation used per copy, for reference

        n_skipped = 0
        for lf in label_files:
            base = os.path.basename(lf)[:-len("_label.npy")]
            face_file = os.path.join(face_dir, base + "_face.npy")
            assert os.path.exists(face_file), "missing face file: {}".format(face_file)

            data = np.load(lf).astype(np.float64)   # (N, 9)
            faces = np.load(face_file).astype(np.int64)  # (F, 3)

            verts = torch.tensor(data[:, :3]).float()
            normals = torch.tensor(data[:, 3:6]).float()
            curv = torch.tensor(data[:, 6:8]).float()
            labels = torch.tensor(data[:, 8]).long()  # 0/1
            faces_t = torch.tensor(faces).long()

            # If augmentation is enabled, first check whether this mesh's n_aug copies
            # already have their operators cached on disk. If so, skip this mesh entirely.
            if self.n_aug > 1:
                cached = _copy_ops_cached_on_disk(
                    base, verts, normals, faces_t, self.n_aug, self.op_cache_dir, self.k_eig, seed=seed)
                if cached >= self.n_aug:
                    n_skipped += 1
                    print("  skip {} : {} copies already cached in op_cache".format(base, cached))
                    continue

            for copy_idx in range(self.n_aug):
                if copy_idx == 0:
                    R = np.eye(3)  # original orientation
                else:
                    rs = np.random.RandomState(seed=(seed + _stable_seed(base + "_copy{}".format(copy_idx))) % (2**31))
                    R = random_rotation_matrix(randgen=rs)

                R_t = torch.from_numpy(R).float()

                # rotate coordinates AND normals by the same matrix
                v = torch.matmul(verts, R_t)
                n = torch.matmul(normals, R_t)

                # center and unit scale positions
                v = diffusion_net.geometry.normalize_positions(v)

                self.verts_list.append(v)
                self.faces_list.append(faces_t)
                self.labels_list.append(labels)
                self.normals_list.append(n)
                self.curv_list.append(curv)
                self.rot_list.append(R)

        if n_skipped > 0:
            print("augmentation disk-cache skip: {} meshes already fully cached (removed from dataset)".format(n_skipped))

        self.N = len(self.verts_list)
        # LRU in-memory operator cache: {idx: (frames, mass, L, evals, evecs, gradX, gradY)}
        self._op_cache = OrderedDict()
        self._op_lock = threading.Lock()

    def __len__(self):
        return len(self.verts_list)

    def _compute_ops(self, i):
        # Compute (or load from disk cache) the geometric operators for sample i.
        outputs = diffusion_net.geometry.get_operators(
            self.verts_list[i], self.faces_list[i], k_eig=self.k_eig, op_cache_dir=self.op_cache_dir)
        return tuple(outputs[0:7])  # frames, mass, L, evals, evecs, gradX, gradY

    def _get_ops(self, i):
        # Thread-safe LRU access to the operator cache.
        with self._op_lock:
            if i in self._op_cache:
                self._op_cache.move_to_end(i)   # mark as most-recently-used
                return self._op_cache[i]
            # Not in cache -> compute (may hit disk cache), then insert
            ops = self._compute_ops(i)
            self._op_cache[i] = ops
            self._op_cache.move_to_end(i)
            # Evict oldest entries to keep RAM bounded
            while len(self._op_cache) > self.max_op_cache:
                self._op_cache.popitem(last=False)
            return ops

    def __getitem__(self, idx):
        frames, mass, L, evals, evecs, gradX, gradY = self._get_ops(idx)
        return (self.verts_list[idx], self.faces_list[idx],
                frames, mass, L, evals, evecs, gradX, gradY,
                self.labels_list[idx],
                self.normals_list[idx], self.curv_list[idx])
