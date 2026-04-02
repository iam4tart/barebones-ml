// octree - tree data structure (hierarchically) used to partition 3d space
// each node - cube (region of space)
// each cube can be subdivided into 8 smaller cubes 'oct'
// each child node corresponds to one of those 8 sub-regions
// it's a 3d analogue to quadtree (2d space into 4 regions)

// allows to quickly find which objects/points lie in a region
// collision detection is easy between nearby objects
// graphics engine use octrees to manage visibility and level of detail

// my use-case: organize 3d points for fast search and compression

#pragma once

#include <iostream>
#include <vector>
#include <array>
#include <memory>
#include <queue>
#include <cmath>
#include <limits>
#include <fstream>

struct Point {
    float x, y, z;
};

struct Neighbor {
    float dist;
    Point point;

    // implicit operator overload for std::less<T>
    bool operator<(const Neighbor& other) const {
        return dist < other.dist; // max-heap
    }
};

struct BoundingBox {
    float x_min, x_max;
    float y_min, y_max;
    float z_min, z_max;

    // if a point lies inside the box
    // we store range, we can derive points using the range if needed
    bool contains(const Point& p) const {
        return (
            p.x >= x_min && p.x <= x_max &&
            p.y >= y_min && p.y <= y_max &&
            p.z >= z_min && p.z <= z_max
        );
    }

    // separating axis test
    // if the cube A is completely left of B or B is completely left to A
    // there is no overlap, else there is
    bool intersects(const BoundingBox& other) const {
        return !(
            x_max < other.x_min || other.x_max < x_min ||
            y_max < other.y_min || other.y_max < y_min ||
            z_max < other.z_min || other.z_max < z_min
        );
    }
};

struct OctreeNode {
    BoundingBox boundary; // cube region
    std::vector<Point> points; // points/actual data stored inside the cube

    // using smart pointers (RAII)
    // 8 child nodes (sub-cubes)
    std::array<std::unique_ptr<OctreeNode>, 8> children;

    OctreeNode(const BoundingBox& box) : boundary(box) {}

    // no need for custom destructor - unique_ptr handles cleanup
};

void subdivide(OctreeNode* node) {
    float xMid = (node->boundary.x_min + node->boundary.x_max) / 2.0f;
    float yMid = (node->boundary.y_min + node->boundary.y_max) / 2.0f;
    float zMid = (node->boundary.z_min + node->boundary.z_max) / 2.0f;

    // create 8 child bounding boxes
    BoundingBox boxes[8] = {
        {node->boundary.x_min, xMid, node->boundary.y_min, yMid, node->boundary.z_min, zMid}, // bottom-left-front
        {xMid, node->boundary.x_max, node->boundary.y_min, yMid, node->boundary.z_min, zMid}, // bottom-right-front
        {node->boundary.x_min, xMid, yMid, node->boundary.y_max, node->boundary.z_min, zMid}, // top-left-front
        {xMid, node->boundary.x_max, yMid, node->boundary.y_max, node->boundary.z_min, zMid}, // top-right-front
        {node->boundary.x_min, xMid, node->boundary.y_min, yMid, zMid, node->boundary.z_max}, // bottom-left-back
        {xMid, node->boundary.x_max, node->boundary.y_min, yMid, zMid, node->boundary.z_max}, // bottom-right-back
        {node->boundary.x_min, xMid, yMid, node->boundary.y_max, zMid, node->boundary.z_max}, // top-left-back
        {xMid, node->boundary.x_max, yMid, node->boundary.y_max, zMid, node->boundary.z_max}  // top-right-back
    };

    // assign children
    for(int i=0; i<8; i++) {
        node->children[i] = std::make_unique<OctreeNode>(boxes[i]);
    }
}

OctreeNode* queryNode(OctreeNode* node, const Point& p) {
    if(!node->boundary.contains(p)) return nullptr;
    if(node->children[0] == nullptr) return node;
    for(int i=0; i<8; i++) {
        OctreeNode* found = queryNode(node->children[i].get(), p);
        if(found) return found;
    }
    return node;
}

// inserting points in octree

// design choice
// higher MAX_POINTS, higher points in a cube, creating more levels slowly
const int MAX_POINTS = 4;

void insert(OctreeNode* node, const Point& p);

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
    // check if point is inside this node's boundary
    if(!node->boundary.contains(p)) return;

    // if node has space to store and no children, store the point
    if(node->points.size() < MAX_POINTS && node->children[0] == nullptr) {
        node->points.push_back(p);
        return;
    }

    // if children don't exist yet, subdivide
    if(node->children[0] == nullptr) {
        subdivide(node);
        // redistribute existing points into children
        for(const auto& oldPoint : node->points) {
            for(int i=0; i<8; i++) {
                if(node->children[i]->boundary.contains(oldPoint)) {
                    // .get() comes from RAII where it gives you
                    // raw pointer without transferring ownership
                    // ownership stays with unique_ptr
                    insert(node->children[i].get(), oldPoint);
                    break;
                }
            }
        }
        node->points.clear();
    }

    // insert new point into the correct child
    for(int i=0; i<8; i++) {
        if(node->children[i]->boundary.contains(p)) {
            insert(node->children[i].get(), p);
            break;
        }
    }
}

// cool fact - time complexity of insertion in octree is closer to O(log n) (depth of tree)
// rather than O(n^3)
// each insertion costs proportional to tree depth
// each subdivision costs proportional to threshold (constant) (MAX_POINTS)

// why -
// when a node subdivides, redistribution happens only for a few points stored in that node
// and not the reshuffling of complete data - just local points
// redistribution is constant, independent of total n
// each insertion starts at root and travels down one path
// for n points, depth grows like O(log n) like a balanced BST
// insertion cost = tree depth * constant redistrbution = O(log n)

// deletion of point also has O(log n) time complexity
bool removePoint(OctreeNode* node, const Point& p) {
    // prune if point is outside the node
    if(!node->boundary.contains(p)) return false;

    // try to remove from this node's point
    for(auto it = node->points.begin(); it != node->points.end(); ++it) {
        // de-referencing from iterator, it gives Point object not points directly - shorthand for (*it).x
        if(it->x == p.x && it->y == p.y && it->z == p.z) {
            node->points.erase(it);
            return true;
        }
    }

    // recurse into children
    // if a point is not inside a node's boundary, it cannot possibly be inside any of its children
    // because children are strictly subdivisions of the parent's cube

    // if the node has been subdivided and has children, loop through all
    if(node->children[0] != nullptr) {
        for(int i=0; i<8; i++) {
            if(removePoint(node->children[i].get(), p)) {
                // after successful deletion, try merging back
                bool allEmpty = true;
                for(int j=0; j<8; j++) {
                    // are all of this node's children completely empty?
                    // if child still has points or if child itself has been subdidivded further meaning
                    // it has grandchildren and is not empty
                    if(node->children[j]->points.size() > 0 || node->children[j]->children[0] != nullptr) {
                        allEmpty = false;
                        break;
                    }
                }
                // cleaning up the structure after the point has been removed
                if(allEmpty) {
                    for(int j=0; j<8; j++) {
                        node->children[j].reset(); // safely deletes and sets nullptr
                    }
                }
                return true;
            }
        }
    }
    return false; // not found
}

// querying or searching for points within a given region (range) - O(log n + k) where k is the number of points found
// usecases: collision detection, visibility checks, spatial filtering
void queryRange(OctreeNode* node, const BoundingBox& range, std::vector<Point>& results) {
    // if node's boundary doesn't intersect query range, skip
    if(!node->boundary.intersects(range)) return;

    // check points stored in this node
    for(const auto& p : node->points) {
        if(range.contains(p)) {
            results.push_back(p);
        }
    }

    // recurse into children if they exist
    if(node->children[0] != nullptr) {
        for(int i=0; i<8; i++) {
            queryRange(node->children[i].get(), range, results);
        }
    }
}

// euclidean distance between two points
float distance(const Point& a, const Point& b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    float dz = a.z - b.z;
    return std::sqrt(dx*dx + dy*dy + dz*dz);
}

// euclidean distance from a query point to a bounding box
// if a point is inside the box, distance is 0
// if a point is outside, how far the point from the nearest face/edge/corner of the box
float distanceToBox(const Point& p, const BoundingBox& box) {
    // {postitive only if point is left of box, point inside box, positive only if point is right of box}
    float dx = std::max({box.x_min-p.x, 0.0f, p.x-box.x_max});
    float dy = std::max({box.y_min - p.y, 0.0f, p.y - box.y_max});
    float dz = std::max({box.z_min - p.z, 0.0f, p.z - box.z_max});
    return std::sqrt(dx*dx + dy*dy + dz*dz);
}

// it also tells us whether a whole cube (subtree) could possibly contain a closer point
// than the best one we've already found

// querying or search the closest point(s) to a query point - average: O(log n), worst: O(n)
// usecases: compression, clustering, pathfinding, similarity search
void nearestNeighbor(const OctreeNode* node, const Point& query, Point& bestPoint, float& bestDist) {

    // if the closest possible distance from the query point to this node's cube
    // is already worse than the best distance so far, skip the entire subtree
    // branch-and-bound pruning
    float boxDist = distanceToBox(query, node->boundary);
    if(boxDist > bestDist) return;

    // if the node is a leaf (or still has points), compute
    for(const auto& p : node->points) {
        float d = distance(query, p);
        if(d < bestDist) {
            bestDist = d;
            bestPoint = p;
        }
    }

    // recurse into children if they exist
    if(node->children[0] != nullptr) {
        for(int i=0; i<8; i++) {
            nearestNeighbor(node->children[i].get(), query, bestPoint, bestDist);
        }
    }
}

// wrapper function
Point findNearestNeighbor(const OctreeNode* root, const Point& query) {
    Point bestPoint{}; // declare and initalize to 0
    // largest finite float [IEEE-754] (~ 3.4e38) better than 1e9 or little higher
    float bestDist = std::numeric_limits<float>::infinity();
    nearestNeighbor(root, query, bestPoint, bestDist);
    return bestPoint;
}

void kNearestNeighbor(const OctreeNode* node, const Point& query, std::priority_queue<Neighbor>& heap, int k) {
    float boxDist = distanceToBox(query, node->boundary);
    if(!heap.empty() && boxDist > heap.top().dist) return; // prune

    // check points in this node
    for(const auto& p : node->points) {
        float d = distance(query, p);
        heap.push({d,p});
        // typecast to int from size_type
        if((int)heap.size() > k) heap.pop(); // keep only k best
    }

    // recurse into children
    if(node->children[0] != nullptr) {
        for(int i=0; i<8; i++) {
            kNearestNeighbor(node->children[i].get(), query, heap, k);
        }
    }
}

// wrapper function
std::vector<Point> findKNearestNeighbor(const OctreeNode* root, const Point& query, int k) {
    std::priority_queue<Neighbor> heap; // max-heap using operator<
    kNearestNeighbor(root, query, heap, k);

    std::vector<Point> result;
    while(!heap.empty()) {
        result.push_back(heap.top().point);
        heap.pop();
    }
    return result;
}

// write tree to stream
void serialize(const OctreeNode* node, std::ostream& os) {
    if(!node) return;

    bool isLeaf = (node->children[0] == nullptr);
    // start at the memory address of isLeaf, treat what you find there
    // as a raw sequence of bytes and copy exactly one byte into the file
    // this tells deserialize if the node contains points or children
    os.write(reinterpret_cast<const char*>(&isLeaf), sizeof(bool));

    if(isLeaf) {
        // node contains points
        size_t numPoints = node->points.size();
        // write total count of points to the binary, tells how much memory to allocate
        os.write(reinterpret_cast<const char*>(&numPoints), sizeof(size_t));
        // go to the start of the actual point in RAM (.data()) and copy entire
        // block of memory (count * 12 bytes per point) to the disk
        // x,y,z - three floats - 12 bytes
        os.write(reinterpret_cast<const char*>(node->points.data()), numPoints*sizeof(Point));
    }
    else {
        // node contains children
        // recurse and visit all 8 children in order (similar to pre-order traversal in BT)
        for(int i=0; i<8; ++i) {
            serialize(node->children[i].get(), os);
        }
    }
}

// reconstruct tree from stream
void deserialize(OctreeNode* node, std::os) {
    bool isLeaf;
    os.read(reinterpret_cast<char*>(&isLeaf), sizeof(bool));

    if(isLeaf) {
        size_t numPoints;
        os.read(reinterpret_cast<char*>(&numPoints), sizeof(size_t));
        // allocate memory and change vector size to numPoints
        node->points.resize(numPoints);
        // copy bits from dile into vector data buffer
        os.read(reinterpret_cast<char*>(node->points.data()), numPoints*sizeof(Point));
    }
    else {
        subdivide(node);
        for(int i=0; i<8; ++i) {
            deserialize(node->children[i].get(), os);
        }
    }
}