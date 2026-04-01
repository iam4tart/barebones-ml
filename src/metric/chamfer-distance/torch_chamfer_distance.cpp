#include <torch/extension.h>
#include <vector>
#include <limits>
#include <cmath>

#include "chamfer_distance.h"

// validate tensor is float32, 2d, on cpu, contiguous
// shape must be [n_points, 3]
void validate_cloud_tensor(const torch::Tensor& t, const string& name) {
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32, got ", t.dtype());
    
}