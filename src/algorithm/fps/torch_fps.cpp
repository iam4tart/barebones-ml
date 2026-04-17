#include <torch/extension.h>
#include "fps.h"

static std::vector<Point> tensor_to_points(torch::Tensor cloud) {
    cloud = cloud.contiguous();

    int N = cloud.size(0);
    auto acc = cloud.accessor<float, 2>();

    std::vector<Point> pts;
    pts.reserve(N);

    for(int i=0; i<N; i++) {
        pts.push_back({acc[i][0], acc[i][1], acc[i][2]});
    }

    return pts;
}

std::tuple<at::Tensor, at::Tensor> fps_torch(at::Tensor cloud, int64_t n_samples, int64_t start_idx) {
    TORCH_CHECK(cloud.dim() == 2, "cloud must be [N, 3]");
    TORCH_CHECK(cloud.size(1) == 3, "cloud must have 3 coordinates");
    
    auto pts = tensor_to_points(cloud);
    auto indices = fps(pts, n_samples, start_idx);

    int K = indices.size();

    auto idx_t = torch::empty({K}, torch::kInt64);
    auto pts_t = torch::empty({K, 3}, torch::kFloat32);

    auto idx_acc = idx_t.accessor<int64_t, 1>();
    auto pts_acc = pts_t.accessor<float, 2>();

    for(int i=0; i<K; i++) {
        int64_t idx = indices[i];
        idx_acc[i] = idx;

        pts_acc[i][0] = pts[idx].x;
        pts_acc[i][1] = pts[idx].y;
        pts_acc[i][2] = pts[idx].z; 
    }

    return {idx_t, pts_t};
}

double mean_coverage_torch(torch::Tensor cloud, torch::Tensor sampled_idx) {
    TORCH_CHECK(sampled_idx.dim() == 1, "indices must be 1D");

    auto pts = tensor_to_points(cloud);

    std::vector<int> idx(sampled_idx.size(0));
    auto acc = sampled_idx.accessor<int64_t, 1>();

    for(int i=0; i < idx.size(); i++) {
        idx[i] = static_cast<int>(acc[i]);
    }

    return static_cast<double>(mean_coverage(pts, idx));
}

double max_coverage_torch(torch::Tensor cloud, torch::Tensor sampled_idx) {
    TORCH_CHECK(sampled_idx.dim() == 1, "indices must be 1D");

    auto pts = tensor_to_points(cloud);

    std::vector<int> idx(sampled_idx.size(0));
    auto acc = sampled_idx.accessor<int64_t, 1>();

    for (int i = 0; i < idx.size(); i++) {
        idx[i] = static_cast<int>(acc[i]);
    }

    return static_cast<double>(max_coverage(pts, idx));
}

TORCH_LIBRARY(barebones_fps, m) {
    m.def("fps(Tensor cloud, int64_t n_samples, int64_t start_idx) -> (Tensor, Tensor)");
    m.def("mean_coverage(Tensor cloud, Tensor idx) -> double");
    m.def("max_coverage(Tensor cloud, Tensor idx) -> double");
}

TORCH_LIBRARY_IMPL(barebones_fps, CPU, m) {
    m.impl("fps", fps_torch);
    m.impl("mean_coverage", mean_coverage_torch);
    m.impl("max_coverage", max_coverage_torch);
}