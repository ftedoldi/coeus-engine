#ifndef SPOTLIGHT_HPP
#define SPOTLIGHT_HPP

#include "Light.hpp"

namespace Odysseus
{
    class SpotLight : public Light
    {
        private:
            Athena::Vector3 _direction;
            float _spotExponent;
            //angle representing the size of the cone where there'll be light
            float _cutOff;
        
        public:
            SpotLight();
            Athena::Vector3 getPosition() const;
            Athena::Vector3 getDirection() const;
            float getCutOff() const;

            virtual void start();
            virtual void update();

            virtual void setOrderOfExecution(const short& newOrderOfExecution);

            virtual short getUniqueID();

            virtual std::string toString();

            void setPosition(Athena::Vector3& position);
            void setDirection(Athena::Vector3& direction);
            void setCutOff(float cutOff);
            void setSpotExponent(float spotExp);

            void setLightShader(Odysseus::Shader* shader) const;

    };
}

#endif