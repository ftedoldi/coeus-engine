#include <Tree.hpp>

namespace Zeus {
    Tree::Tree() {
        root.father = nullptr;
        root.transform = nullptr;
    }

    Tree::Tree(Odysseus::Transform* rootTransform) {
        root.father = nullptr;
        root.transform = rootTransform;
    }

    void Tree::addChild(Node& node) {
        root.children.push_back(node);
    }

    void Tree::addChildTree(Tree* tree) {
        tree->root.father = &root;
        root.children.push_back(tree->root);
    }

    void Tree::deleteChild(const int& i) {
        // if (root.children.size() == 0)
        //     throw std::invalid_argument("EMPTY_TREE::In Tree getChild(name) the tree is currently empty, please fill the Tree before asking for children.");
        
        // if (i < 0 || i > root.children.size())
        //     std::invalid_argument("INDEX_OUT_OF_RANGE::in Tree the index is out of range");

        // delete getChild(i).father;
        // delete getChild(i).transform;
        // getChild(i).children.clear();

        // root.children.erase(root.children.begin() + i);
    }

    void Tree::deleteChild(const std::string& name) {
        // if (root.children.size() == 0)
        //     throw std::invalid_argument("EMPTY_TREE::In Tree getChild(name) the tree is currently empty, please fill the Tree before asking for children.");
    }

    Node& Tree::getChild(const int& i) {
        if (root.children.size() == 0)
            throw std::invalid_argument("EMPTY_TREE::In Tree getChild(name) the tree is currently empty, please fill the Tree before asking for children.");
        
        if (i < 0 || i > root.children.size())
            std::invalid_argument("INDEX_OUT_OF_RANGE::in Tree the index is out of range");

        return root.children[i];
    }

    Node& Tree::getChild(const std::string& name) {
        if (root.children.size() == 0)
            throw std::invalid_argument("EMPTY_TREE::In Tree getChild(name) the tree is currently empty, please fill the Tree before asking for children.");

        // for (int i = 0; i < root.children.size(); i++)
        //     if (root.children[i].transform->name == name)
        //         return root.children[i];
        
        throw std::invalid_argument("INDEX_OUT_OF_RANGE::In Tree getChild(name) could not find any child that has that specified name");
    }
}