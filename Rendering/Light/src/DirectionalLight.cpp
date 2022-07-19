#include "../DirectionalLight.hpp"

namespace Odysseus
{

    DirectionalLight::DirectionalLight()
    {

    }
    Athena::Vector3 DirectionalLight::getDirection() const
    {
        return this->_direction;
    }

    void DirectionalLight::setDirection(Athena::Vector3& dir)
    {
        this->_direction = dir;
    }

    void DirectionalLight::start()
    {

    }
    void DirectionalLight::update()
    {
        this->setLightShader(this->shader);
    }

    void DirectionalLight::setOrderOfExecution(const short& newOrderOfExecution)
    {

    }

    short DirectionalLight::getUniqueID()
    {
        return this->_uniqueID;
    }

    std::string DirectionalLight::toString()
    {
        return "DirectionalLight";
    }

    void DirectionalLight::setLightShader(Odysseus::Shader* shader) const
    {
        shader->use();
        
        shader->setVec3("dirLight.diffuse", this->_diffuse);
        shader->setVec3("dirLight.specular", this->_specular);
        shader->setVec3("dirLight.ambient", this->_ambient);
        shader->setVec3("dirLight.direction", this->_direction);
    }
}
