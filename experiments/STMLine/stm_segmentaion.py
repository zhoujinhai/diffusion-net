"""

# python stm_segmentation.py --input_features xyz_normal_curv --n_aug 4

# 启用 SE 默认
python stm_segmentation.py --input_features xyz_normal_curv --n_aug 2 --se

# 禁用 SE
python stm_segmentation.py --input_features xyz_normal_curv --n_aug 2 -se
"""

import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../../src/"))  # add the path to the DiffusionNet src
import diffusion_net
from stm_dataset import STMDataset

"""
xyz_normal_curv: 96.80%
xyz_normal:      94.38%
xyz:             92.73%
hks:             94.52%

xyz_normal_curv + aug 2 :  94.10%  (改为Z轴旋转, 或者调整训练超参数)
xyz_normal_curv + use_se:  96.14%
"""


# === Options

parser = argparse.ArgumentParser()
parser.add_argument("--evaluate", action="store_true", help="evaluate using the pretrained model")
parser.add_argument("--input_features", type=str, help="features: xyz / xyz_normal / xyz_normal_curv / hks", default="xyz_normal_curv")
parser.add_argument("--n_aug", type=int, default=1, help="number of augmentation copies per training sample (e.g. 4 => 1 sample becomes 4 rotated samples)")
parser.add_argument("--no-se", action="store_false", dest="se", default=True, help="disable Squeeze-and-Excitation channel attention")
args = parser.parse_args()


# system things
device = torch.device("cuda:0")
dtype = torch.float32

# problem/dataset things
n_class = 2

# model
input_features = args.input_features
k_eig = 128

# training settings
train = not args.evaluate
n_epoch = 200
lr = 1e-3
decay_every = 50
decay_rate = 0.5
n_aug = args.n_aug
with_gradient_features = True
use_se = args.se
print("use_se: ", use_se)


# Important paths
base_path = os.path.dirname(__file__)
op_cache_dir = os.path.join(base_path, "data", "op_cache")
model_save_path = os.path.join(base_path, "data/saved_models/stm_seg_{}_4x128.pth".format(input_features))
dataset_path = "/home/heygears/Data/Teeth/STMLines"


# === Load datasets
# NOTE: data augmentation (rotation) is now done INSIDE the dataset at load time:
# each training sample is duplicated into n_aug copies and its geometric operators
# (mass, L, evecs, gradX, gradY, ...) are recomputed and cached for EVERY rotated
# copy, so the operators always match the rotated coordinates. Validation data is
# always kept with n_aug=1 (no augmentation).

test_dataset = STMDataset(dataset_path, train=False, k_eig=k_eig, use_cache=True, op_cache_dir=op_cache_dir)
test_loader = DataLoader(test_dataset, batch_size=None)

if train:
    train_dataset = STMDataset(dataset_path, train=True, k_eig=k_eig, use_cache=True, op_cache_dir=op_cache_dir, n_aug=n_aug)
    train_loader = DataLoader(train_dataset, batch_size=None, shuffle=True)


# === Create the model

C_in={"xyz":3, "xyz_normal":6, "xyz_normal_curv":8, "hks":16}[input_features]

model = diffusion_net.layers.DiffusionNet(C_in=C_in,
                                          C_out=n_class,
                                          C_width=128,
                                          N_block=4,
                                          last_activation=lambda x : torch.nn.functional.log_softmax(x,dim=-1),
                                          outputs_at="vertices",
                                          dropout=True,
                                          with_gradient_features=with_gradient_features,
                                          use_se=use_se)

model = model.to(device)

# === Optimize
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


def build_features(verts, normals, curv, evals, evecs):
    # Build input features from dataset tensors
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


def train_epoch(epoch):

    # Implement lr decay
    if epoch > 0 and epoch % decay_every == 0:
        global lr
        lr *= decay_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    model.train()
    optimizer.zero_grad()

    correct = 0
    total_num = 0
    for data in tqdm(train_loader):

        verts, faces, frames, mass, L, evals, evecs, gradX, gradY, labels, normals, curv = data

        verts = verts.to(device)
        faces = faces.to(device)
        frames = frames.to(device)
        mass = mass.to(device)
        L = L.to(device)
        evals = evals.to(device)
        evecs = evecs.to(device)
        gradX = gradX.to(device)
        gradY = gradY.to(device)
        labels = labels.to(device)
        normals = normals.to(device)
        curv = curv.to(device)

        # NOTE: no per-batch random rotation here -- augmentation is done at
        # dataset level so that operators stay consistent with the vertices.
        features = build_features(verts, normals, curv, evals, evecs)

        preds = model(features, mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY)

        loss = torch.nn.functional.nll_loss(preds, labels)
        loss.backward()

        pred_labels = torch.max(preds, dim=1).indices
        this_correct = pred_labels.eq(labels).sum().item()
        this_num = labels.shape[0]
        correct += this_correct
        total_num += this_num

        optimizer.step()
        optimizer.zero_grad()

    train_acc = correct / total_num
    return train_acc


def test():

    model.eval()

    correct = 0
    total_num = 0
    with torch.no_grad():

        for data in tqdm(test_loader):

            verts, faces, frames, mass, L, evals, evecs, gradX, gradY, labels, normals, curv = data

            verts = verts.to(device)
            faces = faces.to(device)
            frames = frames.to(device)
            mass = mass.to(device)
            L = L.to(device)
            evals = evals.to(device)
            evecs = evecs.to(device)
            gradX = gradX.to(device)
            gradY = gradY.to(device)
            labels = labels.to(device)
            normals = normals.to(device)
            curv = curv.to(device)

            features = build_features(verts, normals, curv, evals, evecs)

            preds = model(features, mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY)

            pred_labels = torch.max(preds, dim=1).indices
            this_correct = pred_labels.eq(labels).sum().item()
            this_num = labels.shape[0]
            correct += this_correct
            total_num += this_num

    test_acc = correct / total_num
    return test_acc


if train:
    print("Training...")
    print("Using n_aug = {} (each training sample duplicated into {} copies)".format(n_aug, n_aug))

    for epoch in range(n_epoch):
        train_acc = train_epoch(epoch)
        test_acc = test()
        print("Epoch {} - Train overall: {:06.3f}%  Test overall: {:06.3f}%".format(epoch, 100*train_acc, 100*test_acc))

    print(" ==> saving last model to " + model_save_path)
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)


test_acc = test()
print("Overall test accuracy: {:06.3f}%".format(100*test_acc))
