import torch
from barebones.libs import fps as _fps

def fps(pts: torch.Tensor, n_samples: int, start_idx: int = 0):
    """
    Returns:
        idx: [K]
        sampled_pts: [K, 3]
    """
    return torch.ops.barebones_fps.fps(pts, n_samples, start_idx)

def mean_coverage(pts: torch.Tensor, idx: torch.Tensor) -> float:
    """
    Lower is better (average coverage)
    """
    return torch.ops.barebones_fps.mean_coverage(pts, idx)

def max_coverage(pts: torch.Tensor, idx: torch.Tensor) -> float:
    """
    Lower is better (worst-case coverage / Hausdorff-like)
    """
    return torch.ops.barebones_fps.max_coverage(pts, idx)