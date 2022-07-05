#include "../PointLight.hpp"

namespace Odysseus
{
    //PointLight::PointLight(){}

    PointLight::PointLight(Athena::Vector3& position, Athena::Vector3& diffuse, Athena::Vector3& specular,
                            Athena::Vector3& ambient, float constant, float linear, float quadratic) :
    Light(diffuse, specular, ambient), _position(position), _constant(constant), _linear(linear), _quadratic(quadratic) {}

    Athena::Vector3 PointLight::getPosition() const
    {
        return this->_position;
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
        this->_position = position;
    }

    void PointLight::setConstant(float& constant)
    {
        this->_constant = constant;
    }

    void PointLight::setLinear(float& linear)
    {
        this->_linear = linear;
    }

    void PointLight::setQuadratic(float& quadratic)
    {
        this->_quadratic = quadratic;
    }

    void PointLight::setLightShader(Odysseus::Shader& shader) const
    {
        shader.setVec3("pointLight.diffuse", this->_diffuse);
        shader.setVec3("pointLight.specular", this->_specular);
        shader.setVec3("pointLight.ambient", this->_ambient);
        shader.setVec3("pointLight.position", this->_position);
        shader.setFloat("pointLight.constant", this->_constant);
        shader.setFloat("pointLight.linear", this->_linear);
        shader.setFloat("pointLight.quadratic", this->_quadratic);
    }
    
}