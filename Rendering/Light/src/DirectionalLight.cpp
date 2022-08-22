#include "../DirectionalLight.hpp"

namespace Odysseus
{

    DirectionalLight::DirectionalLight()
    {
        _ambient = Athena::Vector3();
        _diffuse = Athena::Vector3(0.5f, 0.5f, 0.5f);
        _specular = Athena::Vector3();

        _direction = Athena::Vector3(0.5f, 0.5f, 0.5f).normalized();
        
        shader = new Odysseus::Shader(".\\Shader\\phongShader.vert", ".\\Shader\\phongShader.frag");
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

    int DirectionalLight::getUniqueID()
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
