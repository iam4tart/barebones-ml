import torch
from barebones.libs import chamfer as _chamfer

def chamfer_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.ops.barebones_chamfer.chamfer.chamfer_distance(a, b)