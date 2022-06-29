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

            Component();

            void start();
            void update();

            std::string getUniqueID();

            std::string toString();
    };
}

#endif // __COMPONENT_H__