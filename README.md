# BareBones.ML

## Installation

```bash
git clone https://github.com/iam4tart/barebones-ml
cd barebones
pip install -r requirements.txt
scripts\build_extension.bat
```

## Modules

- [Octree](/docs/octree.md) — contains C++ implementation of Octree (tree data-structure used to partition 3D space) with PyTorch bindings for spatial search and point cloud tasks.
- [Lowest Common Ancestor](/src/algorithm/lca/part_tree.txt) - contains results of LCA query on a hierarchical 3D model structure provided metadata is available.
- [Chamfer Distance](/docs/chamfer_distance.md) - contains C++ implementation of Chamfer Distance Metric commonly used as loss for comparing ground truth vs generated point clouds.