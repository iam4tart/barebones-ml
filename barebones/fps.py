import torch

def fps(pts: torch.Tensor, n_samples: int, start_idx: int = 0):
    """
    returns:
        idx: [k]
        sampled_pts: [k, 3]
    """
    return torch.ops.barebones_fps.fps(pts, n_samples, start_idx)

def mean_coverage(pts: torch.Tensor, idx: torch.Tensor) -> float:
    return torch.ops.barebones_fps.mean_coverage(pts, idx)

def max_coverage(pts: torch.Tensor, idx: torch.Tensor) -> float:
    return torch.ops.barebones_fps.max_coverage(pts, idx)