#include "../PointLight.hpp"

namespace Odysseus
{
    PointLight::PointLight()
    {

    }

    Athena::Vector3 PointLight::getPosition() const
    {
        return this->transform->position;
    }

    float PointLight::getConstant() const
    {
        return this->_constant;
    }

    float PointLight::getLinear() const
    {
        return this->_linear;
    }

    float PointLight::getQuadratic() const
    {
        return this->_quadratic;
    }

    void PointLight::setPosition(Athena::Vector3& position)
    {
        this->transform->position = position;
    }

    void PointLight::setConstant(float constant)
    {
        this->_constant = constant;
    }

    void PointLight::setLinear(float linear)
    {
        this->_linear = linear;
    }

    void PointLight::setQuadratic(float quadratic)
    {
        this->_quadratic = quadratic;
    }

    void PointLight::start()
    {

    }
    void PointLight::update()
    {
        this->setLightShader(this->shader);
    }

    void PointLight::setOrderOfExecution(const short& newOrderOfExecution)
    {

    }

    short PointLight::getUniqueID()
    {
        return this->_uniqueID;
    }

    std::string PointLight::toString()
    {
        return "PointLight";
    }

    void PointLight::setLightShader(Odysseus::Shader* shader) const
    {
        shader->use();
        
        shader->setVec3("pointLight.diffuse", this->_diffuse);
        shader->setVec3("pointLight.specular", this->_specular);
        shader->setVec3("pointLight.ambient", this->_ambient);
        shader->setVec3("pointLight.position", this->transform->position);
        shader->setFloat("pointLight.constant", this->_constant);
        shader->setFloat("pointLight.linear", this->_linear);
        shader->setFloat("pointLight.quadratic", this->_quadratic);
    }
    
}