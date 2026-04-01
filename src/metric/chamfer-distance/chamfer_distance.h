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

// one-directional mean nearest neighbor distance from A to B
// not symmetric on its own
float mean_nn_dist(const vector<Point>& A, const vector<Point>& B) {
    float sum = 0.0f;
    for(const auto& a : A) {
        sum += nearest_sq_dist(a, B);
    }
    return sum / static_cast<float>(A.size());
}

// full chamfer distance between two point clouds
// symmetric: CD(A,B) == CD(B,A)
// time complexity: O(n*m) naive - replace mean_nn_dist with octree for O(n log n)
ChamferResult chamfer_distance(const vector<Point>& A, const vector<Point>& B) {
    ChamferResult result;
    result.mean_ab = mean_nn_dist(A,B);
    result.mean_ba = mean_nn_dist(B,A);
    result.cd = result.mean_ab + result.mean_ba;
    result.cd_x1000 = result.cd * 1000.0f;
    return result;
}

// a named point cloud - maps to one semantic part of 3D model
struct PartCloud {
    string canonical_path;
    vector<Point> points;
};

// per part chamfer distance
// computes CD per semantiic part, not just the whole mesh
// similar to evaluating per-block generation quality
struct PartChamferResult {
    string canonical_path;
    ChamferResult cd_result;
    // if part is not found in generated output
    bool missing;
};

vector<PartChamferResult> per_part_chamfer(const vector<PartCloud>& gt_parts, const vector<PartCloud>& gen_parts) {
    vector<PartChamferResult> results;

    for(const auto& gt : gt_parts) {
        PartChamferResult r;
        r.canonical_path = gt.canonical_path;
        r.missing = true;

        for(const auto& gen : gen_parts) {
            if(gen.canonical_path == gt.canonical_path) {
                r.cd_result = chamfer_distance(gt.points, gen.points);
                r.missing = false;
                break;
            }
        }
        results.push_back(r);
    }
    return results;
}

void print_part_results(const vector<PartChamferResult>& results) {
    cout << "\n--- Per-Part Chamfer Distance (x10^-3) ---\n";
    cout << string(64, '-') << "\n";

    float total_cd = 0.0f;
    int evaluated = 0;

    for(const auto& r : results) {
        if(r.missing) {
            cout << r.canonical_path << ": MISSING\n";
            continue;
        }

        cout << r.canonical_path 
                << " | CD=" << r.cd_result.cd_x1000
                << " | A->B=" << r.cd_result.mean_ab * 1000.0f
                << " | B->A=" << r.cd_result.mean_ba * 1000.0f << "\n";

        total_cd += r.cd_result.cd;
        evaluated++;
    }

    cout << string(64, '-') << "\n";
    if(evaluated > 0) {
        cout << "Mean CD across parts: " << (total_cd / evaluated) * 1000.0f << " x10^-3\n";
    }
}

// helper functions

// sampling points for spherical point cloud using gaussian normalization
// (surface area on sphere) dA = sin(ϕ) dϕ dθ
// the normal distibution is the only distribution where sampling each axis independently
// gives spherical symmetry, it has no directional bias - it's equally likely to point anywhere in 3D space
vector<Point> sample_sphere(int n_points, float radius = 1.0f, unsigned int seed = 42) {
    vector<Point> cloud;
    mt19937 rng(seed);
    // 68% of values fall within -1.0 to 1.0
    normal_distribution<float> dist(0.0f, 1.0f);

    for (int i = 0; i < n_points; i++)
    {
        // sample raw gaussian vector - random direction, random magnitude
        float x = dist(rng);
        float y = dist(rng);
        float z = dist(rng);

        // compute magnitude
        float len = sqrt(x * x + y * y + z * z);

        // normalize to unit sphere
        // now vector has length exactly 1.0
        // direction is preserved, magnitude is discarded
        // and then scale by radius
        cloud.push_back({radius * x / len, radius * y / len, radius * z / len});
    }
    return cloud;
}

// adding gaussian noise to spherical point cloud
// after noise addition, points are nudged and length is changed and they are no longer on sphere surface
vector<Point> add_noise(const vector<Point>& cloud, float sigma = 0.05f, unsigned int seed = 99) {
    vector<Point> noisy_cloud = cloud;
    mt19937 rng(seed);
    // generates random values centered at 0 and spread between [-sigma, +sigma] and far away almost exponentially rare
    normal_distribution<float> dist(0.0f, sigma);
    for(auto &p : noisy_cloud) {
        p.x += dist(rng);
        p.y += dist(rng);
        p.z = dist(rng);
    }
    return noisy_cloud;
}