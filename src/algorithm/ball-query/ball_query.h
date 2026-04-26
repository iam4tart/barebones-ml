// fps selects K centroids
// ball_query finds up to max_samples neighbors per centroid
// pad_group fixes group_size for batching
// normalize_group makes coordinates relative to centroid
// ml-model processes each group into local feature vector using its encoder

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

// find all points within radius of query, capped at max_samples
// return indices
inline std::vector<int> ball_query(const std::vector<Point>& cloud, const Point& query, float radius, int max_samples = -1) {
    int N = static_cast<int>(cloud.size());
    assert(N > 0);

    std::vector<int> neighbors;
    float r2 = radius * radius;

    for(int i=0; i<N; i++) {
        if(sq_dist(cloud[i], query) <= r2) {
            neighbors.push_back(i);

            // stop early when cap is reached
            if(max_samples > 0 && static_cast<int>(neighbors.size()) >= max_samples) break;
        }
    }
    return neighbors;
}

// when ball finds fewer than max_samples neighbors(sparse region)
// pad to exactly max_samples by repeating centroid index (neutral padding)
// keeps tensor shape fixed
// for example, a missing point in sparse region of drone arm would current the local feature
inline std::vector<int> pad_group(const std::vector<int>& group, int max_samples, int query_idx) {
    std::vector<int> padded = group;
    while(static_cast<int>(padded.size()) < max_samples) {
        padded.push_back(query_idx);
    }
    return padded;
}

// to make group coordinates relative to centroids
// so models see the local shape not global position
// subtract centroid from each group point
// for example, two identical motor geometries at different positions in drone
// should produce same local feature (nearly)
// without local normalization, encoder won't nearly even generalize
inline std::vector<Point> normalize_group(const std::vector<Point>& cloud, const std::vector<int>& group_idx, const Point& centroid) {
    std::vector<Point> local;
    local.reserve(group_idx.size());

    for(int idx : group_idx) {
        local.push_back({
            cloud[idx].x - centroid.x,
            cloud[idx].y - centroid.y,
            cloud[idx].z - centroid.z
        });
    }
    return local;
}

// full point set abstraction grouping
inline std::vector<std::vector<Point>> group_points(const std::vector<Point>& cloud, const std::vector<Point>& centroids, const std::vector<int>& centroid_idx, float radius, int max_samples) {
    std::vector<std::vector<Point>> groups;
    groups.reserve(centroids.size());

    for(int k=0; k<static_cast<int>(centroids.size()); k++) {
        auto neighbors = ball_query(cloud, centroids[k], radius, max_samples);
        auto padded = pad_group(neighbors, max_samples, centroid_idx[k]);
        auto local = normalize_group(cloud, padded, centroids[k]);
        groups.push_back(local);
    }
    return groups;
}