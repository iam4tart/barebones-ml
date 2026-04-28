#include <torch/extension.h>
#include "ball_query.h"

static void validate_cloud(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32");
    TORCH_CHECK(t.dim() == 2, name, " must be [N, 3]");
    TORCH_CHECK(t.size(1) == 3, name, " must have 3 columns");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(!t.is_cuda(), name, " must be on cpu");
}

static std::vector<Point> to_cloud(const torch::Tensor& t) {
    std::vector<Point> cloud;
    cloud.reserve(t.size(0));

    const float* d = t.data_ptr<float>();
    for(int64_t i=0; i<t.size(0); i++) {
        cloud.push_back({d[i*3], d[i*3+1], d[i*3+2]});
    }
    return cloud;
}

