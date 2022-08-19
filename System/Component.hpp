#ifndef __COMPONENT_HPP__
#define __COMPONENT_HPP__

#include <SceneObject.hpp>
#include <Transform.hpp>

#include <rttr/registration>

#include <string>
#include <vector>
#include <ctime>

namespace System {

    class Component {
        protected:
            int _uniqueID = 0;
            short _orderOfExecution = 0;

        public:
            Odysseus::SceneObject* sceneObject;
            Odysseus::Transform* transform;

            virtual void start() = 0;
            virtual void update() = 0;

            virtual void setOrderOfExecution(const short& newOrderOfExecution) = 0;

            virtual int getUniqueID() = 0;

            virtual std::string toString() = 0;

            virtual ~Component() {}

            RTTR_ENABLE();
    };

    RTTR_REGISTRATION
    {
        rttr::registration::class_<Component>("Component");
    }
}


#endif // __COMPONENT_HPP__