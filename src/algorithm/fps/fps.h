//  farthest point sampling
//  our usecase - select a subset of points that best represent the entire point cloud
// basically selecting 5 balloons out of 10000 balloons which represents a room

// random sampling may cluster points in one region
// fps tends to spreads points uniformly accross space and produces better coverage of shape with less points

// algorithm:
// pick any starting point usually index 0
// compute distance from every point to the selection set
// pick the point with maximum distance - farthest from everything else
// add it to selection, update distances
// repeat until n sample points selected

// dist[i] = min(dist[i], dist_to_new_added_point)
// no need to compare distance against all selected points
// because dist[i] already stores the min distance to all previous selections

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
std::vector<int> fps(const std::vector<Point>& cloud, int n_samples, int start_idx=0) {
    int N = static_cast<int>(cloud.size());

    assert(N>0);
    assert(n_samples > 0 && n_samples <= N);

    std::vector<float> dist(N, std::numeric_limits<float>::infinity());

    std::vector<int> selected;
    selected.reserve(n_samples);

    selected.push_back(start_idx);

    for(int s=1; s<n_samples; s++) {
        int last = selected.back();

        // if point i is closer to the last than to 
        // any previous selection, update its min distance
        for(int i=0; i<N; i++) {
            float d = sq_dist(cloud[i], cloud[last]);
            if(d < dist[i]) dist[i] = d;
        }

        // point with max distance to selection set
        // this is the point least covered by current selection
        int farthest = 0;
        float max_d = -1.0f;
        for(int i=0; i<N; i++) {
            if(dist[i] > max_d) {
                max_d = dist[i];
                farthest = i;
            }
        }

        // add back to selection the covered point
        selected.push_back(farthest);
    }
    return selected;
}

// return sampled points instead of indices
std::vector<Point> fps_points(const std::vector<Point>& cloud, int n_samples, int start_idx=0) {
    auto indices = fps(cloud, n_samples, start_idx);
    std::vector<Point> result;
    result.reserve(n_samples);
    for(int idx : indices) result.push_back(cloud[idx]);
    return result;
}

// to compare fps vs random sampling quality
float mean_coverage(const std::vector<Point>& sampled, std::vector<Point>& full) {
    float total = 0.0f;
    for(const auto& p : full) {
        float best = std::numeric_limits<float>::infinity();
        for(const auto& s : sampled) {
            float d = sq_dist(p, s);
            if(d<best) best = d;
        }
        total += std::sqrt(best);

        return total/static_cast<float>(full.size());
    }
}