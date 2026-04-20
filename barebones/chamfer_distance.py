import torch

def chamfer_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.ops.barebones_chamfer.chamfer.chamfer_distance(a, b)