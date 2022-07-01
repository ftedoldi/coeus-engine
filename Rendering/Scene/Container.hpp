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
        friend class SceneObject;
        
        private:
            std::vector<Component*> _components;

        public:
            const std::vector<Component*>& components;

            Container(SceneObject& owner, Transform& ownerTransform);

            bool operator == (Container& b);

            ~Container();
    };
}

#endif // __CONTAINTER_H__