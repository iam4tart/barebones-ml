#pragma once

#include <vector>
#include <limits>
#include <random>
#include <cassert>
#include <cmath>

struct Point {
    float x, y, z;
};

// no sqrt needed for comparisons
// inline because it is called millions of times in the algorithm
inline float sq_dist(const Point& a, const Point& b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    float dz = a.z - b.z;
    return dx*dx+dy*dy+dz*dz;
}

// input point cloud, desired number of points to select, index of first point to start sampling
std::vector<int> fps(const std::vector<Point>& cloud, int n_samples, int start_idx=0) {
    int N = static_cast<int>(cloud.size());

    assert(N>0);
    assert(n_samples>0 && n_samples<=N);

    // maintains dist for every point in the cloud to determine which is farthest
    // from exisiting selected set of points during the next iteration
    std::vector<float> dist(N, std::numeric_limits<float>::infinity());
    // points picked so far
    std::vector<int> selected;
    selected.reserve(n_samples);

    // random start if needed
    if(start_idx < 0) {
        std::mt19937 rng(42);
        std::uniform_int_distribution<int> uni(0, N-1);
        start_idx = uni(rng);
    }

    selected.push_back(start_idx);

    for(int s=1; s<n_samples; s++) {
        int last = selected.back();

        // update step
        for(int i=0; i<N; i++) {
            float d = sq_dist(cloud[i], cloud[last]);
            if(d<dist[i]) dist[i] = d;
        }

        // argmax
        int farthest = 0;
        // distances are positives, it will be overwritten at any valid distance
        // this ensures first point becomes the initial farthest candidate
        float max_d = -1.0f;

        for(int i=0; i<N; i++) {
            if(dist[i] > max_d) {
                max_d = dist[i];
                farthest = i;
            }
        }

        selected.push_back(farthest);
    }

    return selected;
}

// extract sampled points from indices
inline std::vector<Point> fps_points(const std::vector<Point>& cloud, int n_samples, int start_idx=0) {
    auto indices = fps(cloud, n_samples, start_idx);
    std::vector<Point> result;
    result.reserve(n_samples);

    for(int idx : indices) result.push_back(cloud[idx]);
    return result;
}

// measure how well the samples represent the original shape
// lower is better because sampled points are spread well
// higher means larger gaps indicating failure in capturing the shape
// less sensitive to outliers
inline float mean_coverage(const std::vector<Point>& pts, const std::vector<int>& sampled_idx) {
    // accumulate gap distance
    float total = 0.0f;

    for(int i=0; i<pts.size(); ++i) {
        float best = std::numeric_limits<float>::infinity();

        // get the best distance from sampled set
        for(int j : sampled_idx) {
            float d = sq_dist(pts[i], pts[j]);
            if(d < best) best = d;
        }

        // actual euclidean distance
        total += std::sqrt(best);
    }

    // density of sampling
    return total/pts.size();
}

// max coverage AKA hausdorff distance measures worst case scenario
// identifies the largest gap means it finds signle most isolated point
// in the original cloud that samples failed to get near
// very sensitive to outliers
inline float max_coverage(const std::vector<Point>& pts, const std::vector<int>& sampled_idx) {
    float worst = 0.0f;

    for(int i=0; i<pts.size(); ++i) {
        float best = std::numeric_limits<float>::infinity();

        for(int j : sampled_idx) {
            float d = sq_dist(pts[i], pts[j]);
            if(d < best) best = d;
        }

        // max filter
        if(best > worst) worst = best;
    }

    return std::sqrt(worst);
}