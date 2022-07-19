#include "../SpotLight.hpp"

namespace Odysseus
{
    SpotLight::SpotLight()
    {

    }

    Athena::Vector3 SpotLight::getPosition() const
    {
        return this->transform->position;
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
        this->transform->position = position;
    }

    void SpotLight::setDirection(Athena::Vector3& direction)
    {
        this->_direction = direction;
    }

    void SpotLight::setCutOff(float cutOff)
    {
        this->_cutOff = cutOff;
    }

    void SpotLight::start()
    {

    }
    void SpotLight::update()
    {
        this->setLightShader(this->shader);
    }

    void SpotLight::setOrderOfExecution(const short& newOrderOfExecution)
    {

    }

    short SpotLight::getUniqueID()
    {
        return this->_uniqueID;
    }

    std::string SpotLight::toString()
    {
        return "SpotLight";
    }

    void SpotLight::setLightShader(Odysseus::Shader* shader) const
    {
        shader->use();
        
        shader->setVec3("spotLight.diffuse", this->_diffuse);
        shader->setVec3("spotLight.specular", this->_specular);
        shader->setVec3("spotLight.ambient", this->_ambient);
        shader->setVec3("spotLight.position", this->transform->position);
        shader->setVec3("spotLight.direction", this->_direction);
        shader->setFloat("spotLight.spotExponent", this->_spotExponent);
        //We calculate the cosine value here because its needed in the fragment shader and also because calculating it in the shader would be expensive
        shader->setFloat("spotLight.cutOff", std::cos(Athena::Math::degreeToRandiansAngle(_cutOff)));
    }
}