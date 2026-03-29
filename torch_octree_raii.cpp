#include <torch/torch.h>
#include <torch/extension.h>
#include <torch/script.h>

#include <iostream>
#include <vector>
#include <array>
#include <memory>
#include <queue>
#include <cmath>
#include <limits>

struct Point {
    float x, y, z;
};

struct Neighbor {
    float dist;
    Point point;
    bool operator<(const Neighbor& other) const {
        return dist < other.dist;
    }
};

struct BoundingBox {
    float x_min, x_max;
    float y_min, y_max;
    float z_min, z_max;

    bool contains(const Point& p) const {
        return (
            p.x >= x_min && p.x <= x_max &&
            p.y >= y_min && p.y <= y_max &&
            p.z >= z_min && p.z <= z_max
        );
    }

    bool intersects(const BoundingBox& other) const {
        return !(
            x_max < other.x_min || other.x_max < x_min ||
            y_max < other.y_min || other.y_max < y_min ||
            z_max < other.z_min || other.z_max < z_min
        );
    }
};

struct OctreeNode {
    BoundingBox boundary;
    std::vector<Point> points;
    std::array<std::unique_ptr<OctreeNode>, 8> children;

    OctreeNode(const BoundingBox& box) : boundary(box) {}
};

const int MAX_POINTS = 4;

void subdivide(OctreeNode* node) {
    float xMid = (node->boundary.x_min + node->boundary.x_max) / 2.0f;
    float yMid = (node->boundary.y_min + node->boundary.y_max) / 2.0f;
    float zMid = (node->boundary.z_min + node->boundary.z_max) / 2.0f;

    BoundingBox boxes[8] = {
        {node->boundary.x_min, xMid, node->boundary.y_min, yMid, node->boundary.z_min, zMid},
        {xMid, node->boundary.x_max, node->boundary.y_min, yMid, node->boundary.z_min, zMid},
        {node->boundary.x_min, xMid, yMid, node->boundary.y_max, node->boundary.z_min, zMid},
        {xMid, node->boundary.x_max, yMid, node->boundary.y_max, node->boundary.z_min, zMid},
        {node->boundary.x_min, xMid, node->boundary.y_min, yMid, zMid, node->boundary.z_max},
        {xMid, node->boundary.x_max, node->boundary.y_min, yMid, zMid, node->boundary.z_max},
        {node->boundary.x_min, xMid, yMid, node->boundary.y_max, zMid, node->boundary.z_max},
        {xMid, node->boundary.x_max, yMid, node->boundary.y_max, zMid, node->boundary.z_max}
    };

    for(int i=0; i<8; i++) {
        node->children[i] = std::make_unique<OctreeNode>(boxes[i]);
    }
}

OctreeNode* queryNode(OctreeNode* node, const Point& p) {
    if(!node->boundary.contains(p)) return nnullptr;
    if(node->children[0] == nullptr) return node;
    for(int i=0; i<8; i++) {
        OctreeNode* found = queryNode(node->children[i].get(), p);
        if(found) return found;
    }
    return node;
}

void redistribute_points(OctreeNode* node) {
    if(node->children[0] == nullptr) return;

    for(const auto& p : node->points) {
        for(int i=0; i<8; i++) {
            if(node->children[i]->boundary.contains(p)) {
                insert(node->children[i].get(), p);
                break;
            }
        }
    }
    node->points.clear();
}

void insert(OctreeNode* node, const Point& p) {
    if(!node->boundary.contains(p)) return;

    if(node->points.size() < MAX_POINTS && node->children[0] == nullptr) {
        node->points.push_back(p);
        return;
    }

    if(node->children[0] == nullptr) {
        subdivide(node);
        for(const auto& oldPoint : node->points) {
            for(int i=0; i<8; i++) {
                if(node->children[i]->boundary.contains(oldPoint)) {
                    insert(node->children[i].get(), oldPoint);
                    break;
                }
            }
        }
        node->points.clear();
    }

    for(int i=0; i<8; i++) {
        if(node->children[i]->boundary.contains(p)) {
            insert(node->children[i].get(), p);
            break;
        }
    }
}

bool removePoint(OctreeNode* node, const Point& p) {
    if(!node->boundary.contains(p)) return false;

    for(auto it = node->points.begin(); it != node->points.end(); ++it) {
        if(it->x == p.x && it->y == p.y && it->z == p.z) {
            node->points.erase(it);
            return true;
        }
    }

    if(node->children[0] != nullptr) {
        for(int i=0; i<8; i++) {
            if(removePoint(node->children[i].get(), p)) {
                bool allEmpty = true;
                for(int j=0; j<8; j++) {
                    if(node->children[j]->points.size() > 0 || node->children[j]->children[0] != nullptr) {
                        allEmpty = false;
                        break;
                    }
                }
                if(allEmpty) {
                    for(int j=0; j<8; j++) {
                        node->children[j].reset();
                    }
                }
                return true;
            }
        }
    }
    return false;
}

void queryRange(OctreeNode* node, const BoundingBox& range, std::vector<Point>& results) {
    if(!node->boundary.intersects(range)) return;

    for(const auto& p : node->points) {
        if(range.contains(p)) {
            results.push_back(p);
        }
    }

    if(node->children[0]) {
        for(int i=0; i<8; i++) {
            queryRange(node->children[i].get(), range, results);
        }
    }
}

float distance(const Point& a, const Point& b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    float dz = a.z - b.z;
    return std::sqrt(dx*dx + dy*dy + dz*dz);
}

float distanceToBox(const Point& p, const BoundingBox& box) {
    float dx = std::max(std::max(box.x_min - p.x, 0.0f), p.x - box.x_max);
    float dy = std::max(std::max(box.y_min - p.y, 0.0f), p.y - box.y_max);
    float dz = std::max(std::max(box.z_min - p.z, 0.0f), p.z - box.z_max);
    return std::sqrt(dx*dx + dy*dy + dz*dz);
}

void nearestNeighbor(const OctreeNode* node, const Point& query, Point& bestPoint, float& bestDist) {
    float boxDist = distanceToBox(query, node->boundary);
    if(boxDist > bestDist) return;

    for(const auto& p : node->points) {
        float d = distance(query, p);
        if(d < bestDist) {
            bestDist = d;
            bestPoint = p;
        }
    }

    if(node->children[0]) {
        for(int i=0; i<8; i++) {
            if(node->children[i]) {
                nearestNeighbor(node->children[i].get(), query, bestPoint, bestDist);
            }
        }
    }
}

Point findNearestNeighbor(const OctreeNode* root, const Point& query) {
    Point bestPoint{};
    float bestDist = std::numeric_limits<float>::infinity();
    nearestNeighbor(root, query, bestPoint, bestDist);
    return bestPoint;
}

void kNearestNeighbor(const OctreeNode* node, const Point& query, std::priority_queue<Neighbor>& heap, int k) {
    float boxDist = distanceToBox(query, node->boundary);
    if(!heap.empty() && boxDist > heap.top().dist) return;

    for(const auto& p : node->points) {
        float d = distance(query, p);
        heap.push({d,p});
        if((int)heap.size() > k) heap.pop();
    }

    if(node->children[0]) {
        for(int i=0; i<8; i++) {
            if(node->children[i]) {
                kNearestNeighbor(node->children[i].get(), query, heap, k);
            }
        }
    }
}

std::vector<Point> findKNearestNeighbor(const OctreeNode* root, const Point& query, int k) {
    std::priority_queue<Neighbor> heap;
    kNearestNeighbor(root, query, heap, k);

    std::vector<Point> result;
    while(!heap.empty()) {
        result.push_back(heap.top().point);
        heap.pop();
    }
    return result;
}

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
    for(int i=0; i<pts.size(0); i++) {
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
    vector<Point> pts{best};
    return pointsToTensor(pts);
}

torch::Tensor k_nearest_neighbor(torch::Tensor query, int k) {
    auto acc = query.accessor<float,1>();
    Point q{acc[0], acc[1], acc[2]};
    std::vector<Point> pts = findKNearestNeighbor(&root, q, k);
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

TORCH_LIBRARY(libtorch_octree_raii, m) {
    m.def("insert_points(Tensor pts) -> ()", insert_points);
    m.def("query_range(Tensor box) -> Tensor", query_range);
    m.def("nearest_neighbor(Tensor query) -> Tensor", nearest_neighbor);
    m.def("k_nearest_neighbor(Tensor query, int k) -> Tensor", k_nearest_neighbor);
    m.def("remove_point(Tensor pt) -> bool", remove_point);
    m.def("subdivide() -> ()", force_subdivide);
    m.def("redistribute() -> ()", redistribute_root);
    m.def("redistribute_query(Tensor query) -> bool", redistribute_query);
}