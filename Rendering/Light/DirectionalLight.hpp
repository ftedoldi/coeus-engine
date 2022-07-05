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

            DirectionalLight(Athena::Vector3& direction, Athena::Vector3& diffuse, Athena::Vector3& specular, Athena::Vector3& ambient);

            Athena::Vector3 getDirection() const;
            void setDirection(Athena::Vector3& dir);

            void setLightShader(Odysseus::Shader& shader) const;

    };
}

#endif