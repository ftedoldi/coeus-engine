#ifndef TREE_HPP
#define TREE_HPP

#include <Transform.hpp>

#include <memory>
#include <vector>
#include <stdexcept>

namespace Zeus {

    namespace Odysseus {
        class Transform;
    }
    
    struct Node
    {
        Node* father;
        Odysseus::Transform* transform;
        std::vector<Node> children;
    };
    
    class Tree {
        public:
            Node root;

            Tree();
            Tree(Odysseus::Transform* rootTransform);

            void addChild(Node& node);
            void addChildTree(Tree* tree);

            void deleteChild(const int& i);
            void deleteChild(const std::string& name);

            Node& getChild(const int& i);
            Node& getChild(const std::string& name);

            ~Tree();
    };
}

#endif