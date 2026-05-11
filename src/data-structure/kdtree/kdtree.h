#pragma once

#include <iostream>
using namespace std;

struct Point {
    int x, y;

    bool operator==(const Point& b) const {
        return x == b.x && y == b.y;
    }
};

struct Node {
    Point p;
    Node* left;
    Node* right;
    char level;

    Node(Point q, char lvl = 'x') {
        p = q;
        left = nullptr;
        right = nullptr;
        level = lvl;
    }
};

Node* insert(Node* root = nullptr, Point q, char lvl = 'x') {
    if(root == nullptr)
        return new Node(q, lvl);
    else {
        if(root->level == 'x') {
            if(q.x < root->p.x)
                root->left = insert(root->left, q, 'y');
            else
                root->right = insert(root->right, q, 'y');
        }
        else if(root->level == 'y') {
            if(q.y < root->p.y)
                root->left = insert(root->left, q, 'x');
            else
                root->right = insert(root->right, q, 'x');
        }
    }

    return root;
}

bool exact_search(Node* root, Point query) {
    if(root == nullptr)
        return false;

    if(root->p == query)
        return true;

    if(root->level == 'x') {
        if(query.x < root->p.x)
            return exact_search(root->left, query);
        else
            return exact_search(root->right, query);
    }
    else {
        if(query.y < root->p.y)
            return exact_search(root->left, query);
        else
            return exact_search(root->right, query);
    }
}

Node* findMin(Node* root, char lvl) {
    if(root == nullptr) return nullptr;

    Node* min = root;

    if(root->level == lvl) {
        if(root->left == nullptr) {
            return root;
        }
        return findMin(root->left, lvl);
    }
    
    Node* leftMin = findMin(root->left, lvl);
    Node* rightMin = findMin(root->right, lvl);

    if(leftMin && leftMin->p.x < min->p.x) min = leftMin;
    if(rightMin && rightMin->p.x < min->p.x) min = rightMin;

    return min;
}

// EDIT: i have to add pruning because we can skip whole subtree


Node* remove(Node* root, Point q) {

    return root;
}