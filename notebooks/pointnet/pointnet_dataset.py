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

