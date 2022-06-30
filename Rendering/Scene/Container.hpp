#ifndef __CONTAINTER_H__
#define __CONTAINTER_H__

#include <Component.hpp>
#include <SceneObject.hpp>
#include <Transform.hpp>

#include <vector>
#include <memory>

namespace Odysseus {
    class Component;
    class SceneObject;
    class Transform;

    class Container {
        private:
            std::vector<Component*> _components;
            SceneObject& _owner;
            Transform& _ownerTransform;

        public:
            const std::vector<Component*>& components;

            Container(SceneObject& owner, Transform& ownerTransform);

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
            Component* getComponent(Component& component);
            void removeComponent(Component& component);

            bool operator == (Container& b);

            ~Container();
    };
}

#endif // __CONTAINTER_H__