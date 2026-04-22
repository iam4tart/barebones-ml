#include <torch/extension.h>
#include <torch/library.h>
#include <vector>
#include <string>
#include <limits>
#include <cmath>

#include "chamfer_distance.h"

// validate tensor is float32, 2d, on cpu, contiguous
// shape must be [n_points, 3]
void validate_cloud_tensor(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32, got ", t.dtype());
    TORCH_CHECK(t.dim() == 2, name, " must be 2d [n_points, 3], got ", t.dim(), "d");
    TORCH_CHECK(t.size(1) == 3, name, " must have 3 columns (x,y,z), got ", t.size(1));
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous - call .contiguous() before passing");
    TORCH_CHECK(!t.is_cuda(), name, " must be on cpu - this extension is cpu-only");
}

// convert [n, 3] float tensor to vector<Point>
std::vector<Point> tensor_to_cloud(const torch::Tensor& t) {
    std::vector<Point> cloud;
    // t.size(0) = n_points
    // cloud.reserve() expects size_t : happens implicitly
    cloud.reserve(t.size(0));
    // get raw pointer to underlying memory
    const float* data = t.data_ptr<float>();
    for(int64_t i=0; i<t.size(0); i++) {
        cloud.push_back({data[i*3], data[i*3+1], data[i*3+2]});
    }
    return cloud;
}

// convert vector<Point> to [n, 3] float tensor
torch::Tensor cloud_to_tensor(const std::vector<Point>& cloud) {
    // create tensor in contiguous memory layout
    // cloud.size() return size_t (unsigned 64 bit) while pytorch expects int64_t
    auto t = torch::zeros({static_cast<int64_t>(cloud.size()), 3}, torch::kFloat32);
    float* data = t.data_ptr<float>();
    for(size_t i=0; i<cloud.size(); i++) {
        data[i*3] = cloud[i].x;
        data[i*3+1] = cloud[i].y;
        data[i*3+2] = cloud[i].z;
    }
    return t;
}

// chamfer_distance(a,b) -> dict with cd, mean_ab, mean_ba, cd_x1000
// a: [n, 3] float32 - ground truth point cloud
// b: [m, 3] float32 - generated point cloud
// returs python dict matching ChamferResult fields
torch::Tensor chamfer_distance_op(const torch::Tensor& a, const torch::Tensor& b) {
    validate_cloud_tensor(a, "a");
    validate_cloud_tensor(b, "b");

    auto cloud_a = tensor_to_cloud(a);
    auto cloud_b = tensor_to_cloud(b);

    ChamferResult result = chamfer_distance(cloud_a, cloud_b);

    torch::Tensor out = torch::tensor({
        result.cd,
        result.mean_ab,
        result.mean_ba,
        result.cd_x1000
    }, torch::kFloat32);

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "chamfer distance — cpu";
    m.def("chamfer_distance", &chamfer_distance_op,
          "chamfer distance between two point clouds\n"
          "args: a [n,3] float32, b [m,3] float32\n"
          "returns: tensor [4] = [cd, mean_ab, mean_ba, cd_x1000]",
          py::arg("a"), py::arg("b"));
}