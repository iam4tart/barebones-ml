import torch

def insert_points(pts: torch.Tensor) -> None:
    torch.ops.barebones_octree.insert_points(pts)

def query_range(box: torch.Tensor) -> torch.Tensor:
    return torch.ops.barebones_octree.query_range(box)

def nearest_neighbor(query: torch.Tensor) -> torch.Tensor:
    return torch.ops.barebones_octree.nearest_neighbor(query)

def k_nearest_neighbor(query: torch.Tensor, k: int) -> torch.Tensor:
    return torch.ops.barebones_octree.k_nearest_neighbor(query, k)

def remove_point(pt: torch.Tensor) -> bool:
    return torch.ops.barebones_octree.remove_point(pt)

def subdivide() -> None:
    torch.ops.barebones_octree.subdivide()

def redistribute() -> None:
    torch.ops.barebones_octree.redistribute()

def redistribute_query(query: torch.Tensor) -> bool:
    return torch.ops.barebones_octree.redistribute_query(query)

def save(path: str) -> None:
    torch.ops.barebones_octree.save_octree(path)

def load(path: str) -> None:
    torch.ops.barebones_octree.load_octree(path)