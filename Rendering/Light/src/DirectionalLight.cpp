#include "../DirectionalLight.hpp"

namespace Odysseus
{

    DirectionalLight::DirectionalLight() : 
    Light(Athena::Vector3(0.5f, 0.5f, 0.5f), Athena::Vector3(1.0f, 1.0f, 1.0f), Athena::Vector3(0.2f, 0.2f, 0.2f)), _direction(Athena::Vector3(0.0f, -1.0f, 0.0f))
    {}

    DirectionalLight::DirectionalLight(Athena::Vector3& direction, Athena::Vector3& diffuse, Athena::Vector3& specular, Athena::Vector3& ambient)
    : Light(diffuse, specular, ambient), _direction(direction)
    {}

    Athena::Vector3 DirectionalLight::getDirection() const
    {
        return this->_direction;
    }

    void DirectionalLight::setDirection(Athena::Vector3& dir)
    {
        this->_direction = dir;
    }

    void DirectionalLight::setLightShader(Odysseus::Shader& shader) const
    {
        shader.setVec3("dirLight.diffuse", this->_diffuse);
        shader.setVec3("dirLight.specular", this->_specular);
        shader.setVec3("dirLight.ambient", this->_ambient);
        shader.setVec3("dirLight.direction", this->_direction);
    }
}
