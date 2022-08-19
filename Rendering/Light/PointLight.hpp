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
            
            Athena::Vector3 getPosition() const;
            float getConstant() const;
            float getLinear() const;
            float getQuadratic() const;

            void setPosition(Athena::Vector3& position);
            void setConstant(float constant);
            void setLinear(float linear);
            void setQuadratic(float quadratic);

            void setLightShader(Odysseus::Shader* shader) const;

            RTTR_ENABLE(Light);
    };

    SERIALIZABLE_FIELDS
    {
        System::Serializable::SerializableClass::serialize<PointLight>()
        .constructor<>()(rttr::policy::ctor::as_raw_ptr)
        .property("_constant", &PointLight::_constant)
        .property("_linear", &PointLight::_linear)
        .property("_quadratic", &PointLight::_quadratic)
        .property("_diffuse", &PointLight::_diffuse)
        .property("_specular", &PointLight::_specular)
        .property("_ambient", &PointLight::_ambient)
        .property("shader", &PointLight::shader);
    }
}


#endif