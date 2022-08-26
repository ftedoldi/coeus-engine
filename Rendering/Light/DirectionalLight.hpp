#ifndef DIRECTIONALLIGHT_HPP
#define DIRECTIONALLIGHT_HPP

#include "Light.hpp"

namespace Odysseus
{
    class DirectionalLight : public Light
    {
        public:
            Athena::Vector3 _direction;

            DirectionalLight();

            Athena::Vector3 getDirection() const;
            void setDirection(Athena::Vector3& dir);

            virtual void start();
            virtual void update();

            virtual void setOrderOfExecution(const short& newOrderOfExecution);

            virtual int getUniqueID();

            virtual std::string toString();

            virtual void showComponentFieldsInEditor();

            virtual void serialize(YAML::Emitter& out);
            virtual System::Component* deserialize(YAML::Node& node);

            void setLightShader(Odysseus::Shader* shader) const;
                        
            SERIALIZABLE_CLASS(System::Component, Light);
    };
}

#endif