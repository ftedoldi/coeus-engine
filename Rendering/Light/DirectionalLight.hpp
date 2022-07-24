#ifndef DIRECTIONALLIGHT_HPP
#define DIRECTIONALLIGHT_HPP

#include "Light.hpp"

namespace Odysseus
{
    class DirectionalLight : public Light
    {
        private:
            Athena::Vector3 _direction;

        public:
            DirectionalLight();

            Athena::Vector3 getDirection() const;
            void setDirection(Athena::Vector3& dir);

            virtual void start();
            virtual void update();

            virtual void setOrderOfExecution(const short& newOrderOfExecution);

            virtual short getUniqueID();

            virtual std::string toString();

            void setLightShader(Odysseus::Shader* shader) const;

    };
}

#endif