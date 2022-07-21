#ifndef POINTLIGHT_HPP
#define POINTLIGHT_HPP

#include <Camera.hpp>

#include "Light.hpp"

namespace Odysseus
{
    class PointLight : public Light
    {
        private:
            float _constant;
            float _linear;
            float _quadratic;
        
        public:
            PointLight();

            virtual void start();
            virtual void update();

            virtual void setOrderOfExecution(const short& newOrderOfExecution);

            virtual short getUniqueID();

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
    };
}

#endif