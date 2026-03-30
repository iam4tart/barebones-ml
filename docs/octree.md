# Octree Torch Extension – Quick Guide

## Conventions
- **Structural ops** → always succeed, return `void`.  
- **Conditional ops** → may fail, return `bool`.  
- **Query ops** → return `Tensor` with results.

## Functions & Expected I/O

| Function | Input Shape | Output | Purpose |
|----------|-------------|--------|---------|
| `insert_points(Tensor pts)` | `[N,3]` | `void` | Insert N points into the octree. |
| `query_range(Tensor box)` | `[6]` (`x_min,x_max,y_min,y_max,z_min,z_max`) | `[M,3]` | Return all points inside the box. |
| `nearest_neighbor(Tensor query)` | `[3]` (`x,y,z`) | `[1,3]` | Return the closest point to query. |
| `k_nearest_neighbor(Tensor query, int k)` | `[3]` + `k` | `[k,3]` | Return k nearest points. |
| `remove_point(Tensor pt)` | `[3]` | `bool` | Remove point if found (`true`), else `false`. |
| `subdivide()` | none | `void` | Force subdivision of root node. |
| `redistribute()` | none | `void` | Push root’s points down into children. |
| `redistribute_query(Tensor query)` | `[3]` | `bool` | Redistribute the node containing the query point. |


## Usage Examples (Python)

```python
import torch, libtorch_octree_raii
```

```python
# Insert points
pts = torch.tensor([[0.1,0.2,0.3],[0.5,0.5,0.5]], dtype=torch.float32)
torch.ops.libtorch_octree_raii.insert_points(pts)
```

```python
# Query range
box = torch.tensor([-1,1,-1,1,-1,1], dtype=torch.float32)
results = torch.ops.libtorch_octree_raii.query_range(box)
```

```python
# Nearest neighbor
q = torch.tensor([0.2,0.2,0.2], dtype=torch.float32)
nn = torch.ops.libtorch_octree_raii.nearest_neighbor(q)
```

```python
# Remove point
removed = torch.ops.libtorch_octree_raii.remove_point(torch.tensor([0.5,0.5,0.5]))
```

```python
# Subdivide + redistribute
torch.ops.libtorch_octree_raii.subdivide()
torch.ops.libtorch_octree_raii.redistribute()
```

```python
# Redistribute specific node
torch.ops.libtorch_octree_raii.redistribute_query(q)
```


# Future Work

## ⚡ Performance Improvements
- [ ] **Batch insertion**: handle multiple points in one traversal to reduce recursion overhead.  
- [ ] **Priority traversal in nearest neighbor**: check closest child first and prune others early.  
- [ ] **Lazy subdivision**: subdivide nodes only when strictly necessary.  

## 🛡️ Error Handling
- [ ] **Validate tensor shapes**: enforce `[N,3]`, `[6]`, `[3]` inputs with clear error messages.  
- [ ] **Out‑of‑bounds points**: report skipped points or optionally expand root boundary.