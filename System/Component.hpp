#pragma once

#ifndef __COMPONENT_HPP__
#define __COMPONENT_HPP__

#include "SceneObject.hpp"
#include <Transform.hpp>

#include <string>
#include <vector>
#include <ctime>

namespace Odysseus {
    class SceneObject;
    class Transform;
}

namespace System {

    class Component {
        protected:
            short _uniqueID = 0;
            short _orderOfExecution =  0;

        public:
            Odysseus::SceneObject* sceneObject;
            Odysseus::Transform* transform;
            
            virtual void start() = 0;
            virtual void update() = 0;

            virtual void setOrderOfExecution(const short& newOrderOfExecution) = 0;

            virtual short getUniqueID() = 0;

            virtual std::string toString() = 0;

            virtual ~Component() {}
    };
}

#endif // __COMPONENT_HPP__