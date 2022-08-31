#ifndef POINTLIGHT_HPP
#define POINTLIGHT_HPP

#include "Light.hpp"

namespace Odysseus
{   
    class PointLight : public Light
    {
        public:
            float _constant;
            float _linear;
            float _quadratic;

            PointLight();

            virtual void start();
            virtual void update();

            virtual void setOrderOfExecution(const short& newOrderOfExecution);

            virtual int getUniqueID();

            virtual std::string toString();

            virtual void showComponentFieldsInEditor();

            virtual void serialize(YAML::Emitter& out);
            virtual System::Component* deserialize(YAML::Node& node);
            
            Athena::Vector3 getPosition() const;
            float getConstant() const;
            float getLinear() const;
            float getQuadratic() const;

            void setPosition(Athena::Vector3& position);
            void setConstant(float constant);
            void setLinear(float linear);
            void setQuadratic(float quadratic);

            void setLightShader(Odysseus::Shader* shader) const;
            void setLightShader(Odysseus::Shader* shader, int index) const;

            ~PointLight();
            
            SERIALIZABLE_CLASS(System::Component, Light);
    };
}


#endif