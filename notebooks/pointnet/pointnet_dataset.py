# modelnet40 dataset loader
# downloads hdf5 fies, returns (points [N, 3], label) per sample
# applies augmentation: random rotation + gaussian jitter

import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import urllib.request
import zipfile

# preprocessed version used in original pointnet paper
# 12311 train / 2468 test samples, 40 categories, 2048 points per shape
MODELNET40_URL = (
    "https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip"
)

def download_modelnet40(root="./data"):
    """download and unzip modelnet40 hdf5 if not already present"""
    zip_path = os.path.join(root, "modelnet40,zip")
    data_path = os.path.join(root, "model40_ply_hdf5_2048")
    
    if os.path.exists(data_path):
        print(f"modelnet40 already at {data_path}")
        return data_path
    
    os.makedirs(root, exist_ok=True)
    print(f"downloading modelnet40 to {zip_path} ...")
    urllib.request.urlretrieve(MODELNET40_URL, zip_path)

    print("extracting ...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(root)

    os.remove(zip_path)
    print(f"done — {data_path}")
    return data_path