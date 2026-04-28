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

// convert groups to tensor
static torch::Tensor groups_to_tensor(const std::vector<std::vector<Point>>& groups) {
    // number of centroids
    int64_t K = static_cast<int64_t>(groups.size());
    // number of points per group (after padding)
    int64_t S = groups.empty() ? 0 : static_cast<int64_t>(groups[0].size());
    auto out = torch::zeros({K, S, 3}, torch::kFloat32);
    // out is contiguous array on disk
    float* dst = out.data_ptr<float>();
    for(int64_t k=0; k<K; k++) {
        for(int64_t s=0; s<S; s++) {
            dst[k*S*3 + s*3 + 0] = groups[k][s].x;
            dst[k*S*3 + s*3 + 1] = groups[k][s].y;
            dst[k*S*3 + s*3 + 2] = groups[k][s].z;
        }
    return out;
    }
}

torch::Tensor ball_query_op(const torch::Tensor& cloud, const torch::Tensor& query, float radius, int max_samples) {
    validate_cloud(cloud, "cloud");
    TORCH_CHECK(query.dim() == 1 && query.size(0) == 3, "query must be [3]");
    TORCH_CHECK(radius > 0.0f, "radius must be > 0");

    auto pts = to_cloud(cloud);
    const float* q = query.contiguous().data_ptr<float>();
    Point query_point = {q[0], q[1], q[2]};
    auto idx = ball_query(pts, query_point, radius, max_samples);
    auto out = torch::zeros({static_cast<int64_t>(idx.size())}, torch::kInt64);
    auto d = out.data_ptr<int64_t>();
    for(size_t i=0; i<idx.size(); i++) d[i] = idx[i];
    return out;
}