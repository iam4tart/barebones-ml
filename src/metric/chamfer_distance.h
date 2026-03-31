// chamfer distance - standard metric for comparing two 3D point clouds
// for each point in cloud A, find its nearest point in cloud B (and vice versa)
// average both directions and sum - symmetric, penalized missing and extra geometry

// my use case: generated mesh point cloud  vs ground truth mesh point cloud

#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <limits>
#include <random>
using namespace std;

struct Point {
    float x, y, z;
};

// a named point cloud - maps to one semantic part of 3D model
struct PartCloud {
    string canonical_path;
    vector<Point> points;
};

// result of a chamfer distance evaludation
struct ChamferResult {
    float cd; // full chamfer distance (mean_ab + mean_ba)
    float mean_ab; // mean nearest-neighbor distance from A to B
    float mean_ba; // mean nearest-neighbor distance from B to A
    float cd_x1000; // scaled by 10^3
};

// squared euclidean distance - avoids sqrt where possible
// used internally for nearest neighbor search
float squared_dist(const Point& a, const Point& b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    float dz = a.z - b.z;
    return dx*dx + dy*dy + dz*dz;
}

// nearest neighbor distance from query point to cloud - naive O(m)
// { might replace with octree nearestNeighbor() for large clouds }
float nearest_sq_dist(const Point& query, const vector<Point>& cloud) {
    float best = numeric_limits<float>::infinity();
    for(const auto& p : cloud) {
        float d = squared_dist(query, p);
        if(d < best) best = d;
    }
    return best;
}

