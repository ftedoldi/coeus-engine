#include "../SpotLight.hpp"

namespace Odysseus
{

    SpotLight::SpotLight() : _position(Athena::Vector3(0.0f, 0.0f, 0.0f)), _direction(Athena::Vector3(0.0f, -1.0f, 0.0f)), _cutOff(20.0f),
                            Light(Athena::Vector3(0.5f, 0.5f, 0.5f), Athena::Vector3(1.0f, 1.0f, 1.0f), Athena::Vector3(0.2f, 0.2f, 0.2f)),
                            _spotExponent(0.0f) {}

    SpotLight::SpotLight(Athena::Vector3& position, Athena::Vector3& diffuse, Athena::Vector3& ambient,
            Athena::Vector3& specular, Athena::Vector3& direction, float cutOff, float spotExponent) :
            Light(diffuse, specular, ambient), _position(position), _direction(direction), _cutOff(cutOff), _spotExponent(spotExponent) {}

    Athena::Vector3 SpotLight::getPosition() const
    {
        return this->_position;
    }

    Athena::Vector3 SpotLight::getDirection() const
    {
        return this->_direction;
    }

    float SpotLight::getCutOff() const
    {
        return this->_cutOff;
    }

    void SpotLight::setPosition(Athena::Vector3& position)
    {
        this->_position = position;
    }

    void SpotLight::setDirection(Athena::Vector3& direction)
    {
        this->_direction = direction;
    }

    void SpotLight::setCutOff(float cutOff)
    {
        this->_cutOff = cutOff;
    }

    void SpotLight::setLightShader(Odysseus::Shader& shader) const
    {
        shader.setVec3("spotLight.diffuse", this->_diffuse);
        shader.setVec3("spotLight.specular", this->_specular);
        shader.setVec3("spotLight.ambient", this->_ambient);
        shader.setVec3("spotLight.position", this->_position);
        shader.setVec3("spotLight.direction", this->_direction);
        shader.setFloat("spotLight.spotExponent", this->_spotExponent);
        //We calculate the cosine value here because its needed in the fragment shader and also because calculating it in the shader would be expensive
        shader.setFloat("spotLight.cutOff", std::cos(Athena::Math::degreeToRandiansAngle(_cutOff)));
    }
}