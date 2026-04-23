#pragma once

#include <vector>
#include <cmath>
#include <limits>
#include <cassert>

struct Point {
    float x, y, z;
};

inline float sq_dist(const Point& a, const Point& b) {
    float dx = a.x-b.x;
    float dy = a.y-b.y;
    float dz = a.z-b.z;
    return dx*dx+dy*dy+dz*dz;
}

// return indices
inline std::vector<int> ball_query(const std::vector<int>& cloud, const Point& query, float radius) {
    int N = static_cast<int>(cloud.size());
    assert(N > 0);

    std::vector<int> neighbors;
    float r2 = radius * radius;

    for(int i=0; i<N; i++) {
        if(sq_dist(cloud[i], query) <= r2) {
            neighbors.push_back(i);
        }
    }
    return neighbors;
}