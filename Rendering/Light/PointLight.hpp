#ifndef POINTLIGHT_HPP
#define POINTLIGHT_HPP
#include "Light.hpp"

namespace Odysseus
{
    class PointLight : public Light
    {
        private:
            Athena::Vector3 _position;
            float _constant;
            float _linear;
            float _quadratic;
        
        public:
            //PointLight();
            
            PointLight(Athena::Vector3& position, Athena::Vector3& diffuse, Athena::Vector3& ambient,
                       Athena::Vector3& specular, float constant, float linear, float quadratic);
            
            Athena::Vector3 getPosition() const;
            float getConstant() const;
            float getLinear() const;
            float getQuadratic() const;

            void setPosition(Athena::Vector3& position);
            void setConstant(float& constant);
            void setLinear(float& linear);
            void setQuadratic(float& quadratic);

            void setLightShader(Odysseus::Shader& shader) const;
    };
}

#endif