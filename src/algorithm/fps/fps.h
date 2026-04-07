//  farthest point sampling
//  our usecase - select a subset of points that best represent the entire point cloud

// random sampling may cluster points in one region
// fps tends to spreads points uniformly accross space and produces better coverage of shape with less points

#pragma once

#include <iostream>
#include <vector>
#include <limits>
#include <random>
#include <cassert>

struct Point {
    float x, y, z;
};

float sq_dist(const Point& a, const Point& b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    float dz = a.z - b.z;
    return dx*dx + dy*dy + dz*dz;
}


// time complexity - O(n * k)
std::vector<Point> fps(const std::vector<Point>& points, int k) {
    assert(points.size() > k);

    int n = points.size();
    std::vector<Point> sampled;
    sampled.reserve(k);

    // distances to nearest selected point
    // initialize as max
    std::vector<float> min_dist(n, std::numeric_limits<float>::max());

    // initialize rng 
    std::mt19937 rng(std::random_device{}());
    // define range
    std::uniform_int_distribution<int> dist(0, n-1);
    // random starting point to avoid bias
    int current = dist(rng);

    for(int i=0; i<k; ++i) {
        const Point& p = points[current];
        sampled.push_back(p);

        // update distances
        for(int j=0; j<n; ++j) {
            float d = sq_dist(p, points[j]);
            if(d < min_dist[j]) {
                min_dist[j] = d;
            }
        }

        // pick farthest point
        float max_dist = -1.0f;
        for(int j=0; j<n; ++j) {
            if(min_dist[j] > max_dist) {
                max_dist = min_dist[j];
                current = j;
            }
        }
    }

    return sampled;
}