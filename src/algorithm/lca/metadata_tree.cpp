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
    Node* lca(const string& path_a, string& path_b) {
        if(!path_to_node.count(path_a)) {
            cerr << "ERROR: path not found: " << path_a << "\n";
            return nullptr;
        }

        if(!path_to_node.count(path_b)) {
            cerr << "ERROR: path node found: " << path_b << "\n"; 
            return nullptr;
        }

        Node* a = path_to_node[path_a];
        Node* b = path_to_node[path_b];


    }
};
