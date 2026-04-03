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
    "https://huggingface.co/datasets/Msun/modelnet40/blob/main/modelnet40_ply_hdf5_2048.zip"
)

def download_modelnet40(root="./datasets"):
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

def load_h5_files(data_path, split="train"):
    """
    load all hdf5 files for a split
    each file contains up to 2048 samples
    returns concatenated points [N, 2048, 3] and labels [N]
    """
    list_file = os.path.join(data_path, f"{split}_files.txt")
    with open(list_file, 'r') as f:
        files = [
            os.path.join(data_path, os.path.basename(line.strip())) for line in f.readlines()
        ]
        
    all_points = []
    all_labels = []
    
    for fpath in files:
        with h5py.File(fpath, 'r') as h5:
            points = h5['data'][:] # [n, 2048, 3]
            labels = h5['label'][:] # [n, 1]
            all_points.append(points)
            all_labels.append(labels.squeeze(1))
            
    all_points = np.concatenate(all_points, axis=0) # [total_n, 2048, 3]
    all_labels = np.concatenate(all_labels, axis=0) # [total_n]
    return all_points, all_labels

def random_point_sample(points, num_points):
    """randomly sample num_points frmo a point cloud without replacement"""
    n = points.shape[0]
    if n >= num_points:
        idx = np.random.choice(n, num_points, replace=False)
    else:
        # if fewer points than requested, sample with replacement
        idx = np.random.choice(n, num_points, replace=True)
    return points[idx]

def normalize_point_cloud(points):
    """
    center to origin and scale to unit sphere
    same preprocessing as original pointnet paper
    
    points [N, 3] - each row is one points (x,y,z)
    """
    centroid = np.mean(points, axis=0)
    # cnetering the point cloud around origin
    points = points - centroid
    # furthest point distance from origin
    scale = np.max(np.sqrt(np.sum(points**2, axis=1)))
    points = points / scale
    return points

def random_rotate_y(points):
    """
    random rotation around y-axis
    modelnet40 objects are upright so only y rotation makes sense
    """
    theta = np.random.uniform(0, 2*np.pi)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    # rotation matrix around y-axis (i.e y stays same and rotation in x-z plane)
    R = np.array([
        [cos_t, 0, sin_t],
        [0, 1, 0],
        [-sin_t, 0, cos_t]
    ])
    return points @ R.T

def random_jitter(points, sigma=0.01, clip=0.05):
    """
    add gaussian noise to each point
    params from original pointnet paper
    
    generate noise and cut extreme values
    sidenote - *points is spread operator for destructuring or argument unpacking
    """
    noise = np.clip(sigma * np.random.randn(*points.shape), -clip, clip)
    return points + noise

class ModelNet40(Dataset):
    """
    modelnet40 point cloud classification dataset
    
    returns:
        points: tensor [num_points, 3] - normalized augmented point cloud
        label: tensor [] - integer class label 0-39
    """
    
    def __init__(self,
                 root="./dataset",
                 split="train",
                 num_points=1024,
                 augment=True):
        
        assert split in ("train", "test")
        self.num_points = num_points
        self.augment = augment and (split == "train")
        
        data_path = download_modelnet40(root)
        self.points, self.labels = load_h5_files(data_path, split)
        
        # load class names
        shape_names_files = os.path.join(data_path, "shape_names.txt")
        with open(shape_names_files) as f:
            self.classes = [line.strip() for line in f.readlines()]
            
        print(f"modelnet40 {split}: {len(self.points)} samples, " f"{len(self.classes)} classes")
        
    def __len__(self):
        return len(self.points)
    
    def __getitem__(self, idx):
        pts = self.points[idx]
        label = int(self.labels[idx])
        
        # subsample to num_points
