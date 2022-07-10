#include "../Light.hpp"
namespace Odysseus
{
    Athena::Vector3 Light::getDiffuse() const
    {
        return this->_diffuse;
    }
    Athena::Vector3 Light::getAmbient() const
    {
        return this->_ambient;
    }
    Athena::Vector3 Light::getSpecular() const
    {
        return this->_specular;
    }
    void Light::setDiffuse(Athena::Vector3& diff)
    {
        this->_diffuse = diff;
    }
    void Light::setAmbient(Athena::Vector3& amb)
    {
        this->_ambient = amb;
    }
    void Light::setSpecular(Athena::Vector3& spec)
    {
        this->_specular = spec;
    }
    
    void Light::setShader(Shader* shader)
    {
        this->shader = shader;
    }

}
