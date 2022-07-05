#ifndef SPOTLIGHT_HPP
#define SPOTLIGHT_HPP
#include "Light.hpp"

namespace Odysseus
{
    class SpotLight : public Light
    {
        private:
            Athena::Vector3 _position;
            Athena::Vector3 _direction;
            float _spotExponent;
            //angle representing the size of the cone where there'll be light
            float _cutOff;
        
        public:
            SpotLight();
            SpotLight(Athena::Vector3& position, Athena::Vector3& diffuse, Athena::Vector3& ambient,
                       Athena::Vector3& specular, Athena::Vector3& direction, float cutOff, float spotExponent);
            Athena::Vector3 getPosition() const;
            Athena::Vector3 getDirection() const;
            float getCutOff() const;

            void setPosition(Athena::Vector3& position);
            void setDirection(Athena::Vector3& direction);
            void setCutOff(float cutOff);

            void setLightShader(Odysseus::Shader& shader) const;

    };
}

#endif