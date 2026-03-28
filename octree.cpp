// octree - tree data structure (hierarchically) used to partition 3d space
// each node - cube (region of space)
// each cube can be subdivided into 8 smaller cubes 'oct'
// each child node corresponds to one of those 8 sub-regions
// it's a 3d analogue to quadtree (2d space into 4 regions)

// allows to quickly find which objects/points lie in a region
// collision detection is easy between nearby objects
// graphics engine use octrees to manage visibility and level of detail

// my use-case: organize 3d points for fast search and compression

#include <iostream>
#include <vector>
using namespace std;

struct Point {
    float x, y, z;
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
    vector<Point> points; // points/actual data stored inside the cube
    OctreeNode* children[8] = {nullptr}; // 8 child nodes (sub-cubes)

    OctreeNode(const BoundingBox& box) : boundary(box) {}

    ~OctreeNode() {
        for(int i=0; i<8; i++) {
            delete children[i]; // safe even if nullptr
        }
    }
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
        node->children[i] = new OctreeNode(boxes[i]);
    }
}

// inserting points in octree

// design choice 
// higher MAX_POINTS, higher points in a cube, creating more levels slowly
const int MAX_POINTS = 4;

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
                    insert(node->children[i], oldPoint);
                    break;
                }
            }
        }
        node->points.clear();
    }

    // insert new point into the correct child
    for(int i=0; i<8; i++) {
        if(node->children[i]->boundary.contains(p)) {
            insert(node->children[i], p);
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
            if(removePoint(node->children[i], p)) {
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
                            delete node->children[j];
                            node->children[j] = nullptr;
                        }
                    }
                    return true;
                }
            }
        }
    }

    return false; // not found
}

// querying or searching for points within a given region (range) - O(log n + k) where k is the number of points found
// usecases: collision detection, visibility checks, spatial filtering
void queryRange(const OctreeNode* node, const BoundingBox& range, vector<Point>& results) {
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
            queryRange(node->children[i], range, results);
        }
    }
}

// euclidean distance between two points
float distance(const Point& a, const Point& b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    float dz = a.z - b.z;
    return sqrt(dx*dx + dy*dy + dz*dz);
}

// euclidean distance from a query point to a bounding box
// if a point is inside the box, distance is 0
// if a point is outside, how far the point from the nearest face/edge/corner of the box
float distanceToBox(const Point& p, const BoundingBox& box) {
    float dx = 
}

// it also tells us whether a whole cube (subtree) could possibly contain a closer point
// than the best one we've already found

// querying or search the closest point(s) to a query point
// usecases: compression, clustering, pathfinding, similarity search



// driver code
int main() {
    BoundingBox rootBox{0,10,0,10,0,10};
    OctreeNode root(rootBox);

    insert(&root, Point{1,1,1});
    insert(&root, Point{9,9,9});
    insert(&root, Point{5,5,5});

    BoundingBox region{0,5,0,5,0,5};
    vector<Point> results;
    queryRange(&root, region, results);

    for(auto& p : results) {
        cout << "(" << p.x << "," << p.y << "," << p.z << ")\n";
    }
    return 0;

    // output:
    // (1,1,1)
    // (5,5,5)
}

// to add:
// nearest neighbor search (find closest point)