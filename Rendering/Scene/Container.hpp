#ifndef __CONTAINTER_H__
#define __CONTAINTER_H__

#include <SceneObject.hpp>
#include <Transform.hpp>

#include <vector>
#include <memory>

namespace System {
    class Component;
}

namespace Odysseus {
    class SceneObject;
    class Transform;

    class Container {
        friend class SceneObject;
        
        private:
            std::vector<System::Component*> _components;

        public:
            const std::vector<System::Component*>& components;

            Container(SceneObject& owner, Transform& ownerTransform);

            bool operator == (Container& b);

            ~Container();
    };
}

#endif // __CONTAINTER_H__