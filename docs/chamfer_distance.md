# Chamfer Distance - Quick Guide

A metric that measures how different two 3D point clouds are geometrically.
Given cloud A (ground truth mesh) and cloud B (generated mesh), CD measures
how far apart they are on average — in both directions.

## Formula

$$CD(A, B) = \frac{1}{|A|} \sum_{a \in A} \min_{b \in B} \|a - b\|^2
           + \frac{1}{|B|} \sum_{b \in B} \min_{a \in A} \|b - a\|^2$$

```
CD(A, B) = mean_ab + mean_ba
```

- might add RAII safety to chamfer_distance.h