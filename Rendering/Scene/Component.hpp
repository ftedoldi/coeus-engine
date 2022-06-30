#ifndef __COMPONENT_H__
#define __COMPONENT_H__

#include "SceneObject.hpp"
#include <Transform.hpp>

#include <string>
#include <vector>

namespace Odysseus {
    class SceneObject;
    class Transform;

    class Component {
        protected:
            std::string _uniqueID;

        public:
            SceneObject* sceneObject;
            Transform* transform;

            virtual void start() = 0;
            virtual void update() = 0;

            virtual std::string getUniqueID() = 0;

            virtual std::string toString() = 0;

            virtual ~Component() {}
    };
}

#endif // __COMPONENT_H__