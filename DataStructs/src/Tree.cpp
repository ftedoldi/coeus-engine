#include <Tree.hpp>

namespace Zeus {
    Tree::Tree() {
        root = new Node;

        root->father = nullptr;
        root->transform = nullptr;
    }

    Tree::Tree(Tree* t) {
        this->root = new Node;
        
        this->root = t->root;
    }

    Tree::Tree(Odysseus::Transform* rootTransform) {
        root = new Node;
        
        root->father = nullptr;
        root->transform = rootTransform;
    }

    void Tree::addChild(Node* node) {
        root->children.push_back(node);
    }

    void Tree::addChild(Odysseus::Transform& transform) {
        Node* node = new Node;
        node->father = root;
        node->transform = &transform;

        root->children.push_back(node);
    }

    void Tree::addChildTree(Tree& t) {
        t.root->father = this->root;
        this->root->children.push_back(t.root);
    }

    void Tree::deleteChild(const int& i) {
        if (root->children.size() == 0)
            return;
        
        if (i < 0 || i > root->children.size())
            return;

        delete root->children[i];

        root->children.erase(root->children.begin() + i);
    }

    void Tree::deleteChild(const std::string& name) {
        for (int i = 0; i < root->children.size(); i++)
            if (root->children[i]->transform->name == name) {
                delete root->children[i];
                root->children.erase(root->children.begin() + i);
                return;
            }
    }

    Node* Tree::getChild(const int& i) {
        if (root->children.size() == 0)
            return nullptr;
        
        if (i < 0 || i > root->children.size())
            return nullptr;

        return root->children[i];
    }

    Node* Tree::getChild(const std::string& name) {
        if (root->children.size() == 0)
            return nullptr;

        for (int i = 0; i < root->children.size(); i++)
            if (root->children[i]->transform->name == name)
                return root->children[i];
        
        return nullptr;
    }

    void Tree::deleteTree(Node* n)
    {
        if (n->children.size() == 0) {
            delete n->father;
            delete n->transform;
            return;
        }
        
        for (int i = 0; i < n->children.size(); i++) {
            deleteTree(n->children[i]);
            delete n->children[i];
        }
        
        n->children.clear();
        delete n->father;
        delete n->transform;
    }

    void Tree::deleteTree()
    {
        deleteTree(this->root);
    }

    Tree::~Tree()
    {
        deleteTree(this->root);
    }

}