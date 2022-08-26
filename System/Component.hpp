#ifndef __COMPONENT_HPP__
#define __COMPONENT_HPP__

#include <SceneObject.hpp>
#include <Transform.hpp>

#include <yaml-cpp/yaml.h>
#include <rttr/registration>

#include <string>
#include <vector>
#include <ctime>

namespace System {
    class Component {
        protected:
            int _uniqueID = 0;
            short _orderOfExecution = 0;

            int _editorTextureID = 0;
            bool _hasEditorTexture = false;

        public:
            Odysseus::SceneObject* sceneObject;
            Odysseus::Transform* transform;

            virtual void start() = 0;
            virtual void update() = 0;

            virtual void setOrderOfExecution(const short& newOrderOfExecution) = 0;

            virtual int getUniqueID() = 0;

            virtual std::string toString() = 0;

            // ----------------------------------- Serializable Class Fields ------------------------------------ //
            int getEditorTextureID() { return _editorTextureID; }
            bool hasEditorTexture() { return _hasEditorTexture; }

            virtual void showComponentFieldsInEditor() {}

            virtual void serialize(YAML::Emitter& out) {}
            virtual System::Component* deserialize(YAML::Node& node) { return nullptr; }
            // ------------------------------------------------------------------------------------------------- //

            virtual ~Component() {}

            RTTR_ENABLE();
    };

    RTTR_REGISTRATION
    {
        rttr::registration::class_<Component>("Component");
    }
}


#endif // __COMPONENT_HPP__