#include <torch/torch.h>
#include <torch/extension.h>
// #include <torch/script.h>
#include <iostream>
#include <ostream>

#include "octree_raii.h"

// ------------------ Global Root ------------------
static BoundingBox rootBox{-1,1,-1,1,-1,1};
static OctreeNode root(rootBox);

torch::Tensor pointsToTensor(const std::vector<Point>& pts) {
    std::vector<float> data(pts.size() * 3);
    for(size_t i=0; i<pts.size(); i++) {
        data[i*3 + 0] = pts[i].x;
        data[i*3 + 1] = pts[i].y;
        data[i*3 + 2] = pts[i].z;
    }
    return torch::from_blob(data.data(), {(long)pts.size(), 3}).clone();
}

// accesors gives a lightweight view into the the tensor as a 2d array of floats
// accessors only work if the tensor is continguous in memory, if sliced or transposed
// may need to call .contiguous() first
// it doesnt duplicate or allocate new memory
// it creates a small object that knows pointer to tensor's data, tensor's shape and strides, type and dimensionality

void insert_points(torch::Tensor pts) {
    auto acc = pts.accessor<float,2>();
    for(int64_t i=0; i<pts.size(0); i++) {
        Point p{acc[i][0], acc[i][1], acc[i][2]};
        insert(&root, p);
    }
}

torch::Tensor query_range(torch::Tensor box) {
    auto acc = box.accessor<float,1>();
    BoundingBox bb{acc[0], acc[3], acc[1], acc[4], acc[2], acc[5]};
    std::vector<Point> results;
    queryRange(&root, bb, results);
    return pointsToTensor(results);
}

torch::Tensor nearest_neighbor(torch::Tensor query) {
    auto acc = query.accessor<float,1>();
    Point q{acc[0], acc[1], acc[2]};
    Point best = findNearestNeighbor(&root, q);
    std::vector<Point> pts{best};
    return pointsToTensor(pts);
}

torch::Tensor k_nearest_neighbor(torch::Tensor query, int64_t k) {
    auto acc = query.accessor<float,1>();
    Point q{acc[0], acc[1], acc[2]};
    std::vector<Point> pts = findKNearestNeighbor(&root, q, (int)k);
    return pointsToTensor(pts);
}

bool remove_point(torch::Tensor pt) {
    auto acc = pt.accessor<float,1>();
    Point p{acc[0], acc[1], acc[2]};
    return removePoint(&root, p);
}

void force_subdivide() {
    subdivide(&root);
}

void redistribute_root() {
    redistribute_points(&root);
}

bool redistribute_query(torch::Tensor query) {
    auto acc = query.accessor<float, 1>();
    Point p{acc[0], acc[1], acc[2]};
    OctreeNode* target = queryNode(&root, p);
    if(target) {
        redistribute_points(target);
        return true;
    }
    return false;
}

void save_octree(std::string path) {
    std::ofstream os(path, std::ios::binary);
    serialize(&root, os);
}

void load_octree(std::string path) {
    // load file with binary flag
    std::ifstream is(path, std::ios::binary);
    // delete any existing children to prevent memory leaks or overlapping trees
    for(auto& child : root.children) child.reset();
    // clear the root's points so we do NOT double count them
    root.points.clear();
    deserialize(&root, is);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "octree — cpu";
    m.def("insert_points",      &insert_points,      py::arg("pts"));
    m.def("query_range",        &query_range,        py::arg("box"));
    m.def("nearest_neighbor",   &nearest_neighbor,   py::arg("query"));
    m.def("k_nearest_neighbor", &k_nearest_neighbor, py::arg("query"), py::arg("k"));
    m.def("remove_point",       &remove_point,       py::arg("pt"));
    m.def("subdivide",          &force_subdivide);
    m.def("redistribute",       &redistribute_root);
    m.def("redistribute_query", &redistribute_query, py::arg("query"));
    m.def("save_octree",        &save_octree,        py::arg("path"));
    m.def("load_octree",        &load_octree,        py::arg("path"));
}