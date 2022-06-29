#ifndef __CONTAINTER_H__
#define __CONTAINTER_H__

#include "Component.hpp"

#include <vector>

namespace Odysseus {
    class Container {
        private:
            std::vector<Component *> components;

        public:
            Container();

            // template<class T> void addComponent() {
            //     components.push_back(new T());
            // }

            // template<class T> T* getComponent() {
            //     for (int i = 0; i < components.size(); i++) {
            //         if (components[i]->getUniqueID() == T.getUniqueID())
            //             return (T*)components[i];
            //     }
            // }

            void addComponent(Component& component);
            Component& getComponent(Component& component);
            void removeComponent(Component& component);
    };
}

#endif // __CONTAINTER_H__