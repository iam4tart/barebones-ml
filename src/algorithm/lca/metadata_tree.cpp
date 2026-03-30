#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <algorithm>
using namespace std;

// each node in the metadata tree
struct Node
{
    string name;
    string absolute_path;
    int depth;
    Node *parent;
    vector<Node *> children;

    Node(string name, string absolute_path, int depth, Node *parent) : name(name), absolute_path(absolute_path), parent(parent) {}
};

// tree owns all nodes
struct MetadataTree
{
    Node *root = nullptr;
    unordered_map<string, Node *> path_to_node;

    // insert a full path
    void insert(const string &absolute_path)
    {
        vector<string> parts;
        stringstream ss(absolute_path);
        string token;

        // split by '/'
        while (getline(ss, token, '/'))
        {
            parts.push_back(token);
        }

        string current_path = "";
        Node *current_parent = nullptr;

        for (int i = 0; i < (int)parts.size(); i++)
        {
            current_path += (i == 0 ? "" : "/") + parts[i];

            // node already exists, traverse down
            if (path_to_node.count(current_path))
            {
                current_parent = path_to_node[current_path];
                continue;
            }

            // create new node
            Node *node = new Node(parts[i], current_path, i, current_parent);

            // connect parent and child
            if (current_parent != nullptr)
            {
                current_parent->children.push_back(node);
            }
            else
            {
                root = node; // first node is root
            }

            path_to_node[current_path] = node;
            current_parent = node;
        }
    }

    // load all paths from file
    void load_from_file(const string &filepath)
    {
        ifstream file(filepath);

        if (!file.is_open())
        {
            cerr << "ERROR: cannot open file: " << filepath << "\n";
            return;
        }

        string line;

        while (getline(file, line))
        {
            // skip empty lines and comments
            if (line.empty() || line[0] == '#')
                continue;
            insert(line);
        }
        cout << "loaded " << path_to_node.size() << " nodes from " << filepath << "\n";
    }

    // LCA : lowest common ancestor of two nodes by path
    // naive - O(depth)
    Node* lca(const string& path_a, const string& path_b) {
        if(!path_to_node.count(path_a)) {
            cerr << "ERROR: path not found: " << path_a << "\n";
            return nullptr;
        }

        if(!path_to_node.count(path_b)) {
            cerr << "ERROR: path not found: " << path_b << "\n"; 
            return nullptr;
        }

        Node* a = path_to_node[path_a];
        Node* b = path_to_node[path_b];

        // walk both nodes up to the same depth first
        while(a->depth > b->depth) a = a->parent;
        while(b->depth > a->depth) b = b->parent;

        // now both at same depth, walk up together
        while(a != b) {
            a = a->parent;
            b = b->parent;
        }

        return a;
    }

    // print subtree rooted at a given path (for debug, in YAML format)
    void print_subtree(const string& path, int indent = 0) {
        if(!path_to_node.count(path)) return;

        Node* node = path_to_node[path];
        
    }

    // print stats
    void print_stats() {
        int max_depth = 0;
        int leaf_count = 0;
        
        for(auto& [path, node] : path_to_node) {
            max_depth = max(max_depth, node->depth);
            if(node->children.empty()) leaf_count++;
        }

        cout << "total nodes: " << path_to_node.size() << "\n";
        cout << "max depth: " << max_depth << "\n";
        cout << "leaf nodes: " << leaf_count << "\n";
    }

    // clean heap memory
    ~MetadataTree() {
        for(auto& [path, node] : path_to_node) delete node;
    }
};

void query(MetadataTree& tree, const string& a, const string& b) {
    cout << "\nLCA query:\n";
    cout << " A: " << a << "\n";
    cout << " B: " << b << "\n";
    Node* result = tree.lca(a,b);
    if(result) {
        cout << " LCA: " << result->absolute_path << "\n (depth=" << result->depth << ")\n";
    }
}

int main()
{
    MetadataTree tree;
    tree.load_from_file("metadata_parts.txt");

    std::cout << "\n--- Tree Stats ---\n";
    tree.print_stats();

    std::cout << "\n--- Propulsion Subtree ---\n";
    tree.print_subtree("drone/propulsion");

    query(tree,
          "drone/propulsion/rotor_fl/motor/rotor_bell/magnet_n_1/neodymium_core",
          "drone/propulsion/rotor_fl/motor/stator/winding_a/copper_wire");

    query(tree,
          "drone/propulsion/rotor_fl/propeller/blade_1/tip",
          "drone/propulsion/rotor_fr/propeller/blade_1/tip");

    query(tree,
          "drone/propulsion/rotor_fl/motor/shaft/bearing_top/ball_1",
          "drone/propulsion/rotor_fl/esc/pcb/mosfet_1");

    query(tree,
          "drone/propulsion/rotor_fl/motor/stator/winding_a",
          "drone/sensors/gps/antenna/patch_element");

    query(tree,
          "drone/flight_controller/fc_board/imu/gyroscope/mems_die",
          "drone/flight_controller/fc_board/imu/accelerometer/mems_die");

    query(tree,
          "drone/propulsion/rotor_fl/motor",
          "drone/propulsion/rotor_fl/motor");

    return 0;
}
