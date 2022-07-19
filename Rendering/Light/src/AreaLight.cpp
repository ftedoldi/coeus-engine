#include "../AreaLight.hpp"

namespace Odysseus
{

    AreaLight::AreaLight()
    {

    }
    void AreaLight::start()
    {
        unsigned int numLights = this->pointLights.size();
        shader->setInt("numLights", numLights);
    }
    void AreaLight::update()
    {
        this->setLightShader(this->shader);
    }

    void AreaLight::setOrderOfExecution(const short& newOrderOfExecution)
    {

    }

    short AreaLight::getUniqueID()
    {
        return this->_uniqueID;
    }

    std::string AreaLight::toString()
    {
        return "AreaLight";
    }

    void AreaLight::addLights(PointLight& pt)
    {
        this->pointLights.push_back(pt);
    }

    void AreaLight::setLightShader(Odysseus::Shader* shader) const
    {
        shader->use();
        
        for(unsigned int i = 0; i < this->pointLights.size(); ++i)
        {
            shader->setVec3("pointLights[i].position", pointLights[i].getPosition());
            shader->setVec3("pointLights[i].ambient", pointLights[i].getAmbient());
            shader->setVec3("pointLights[i].diffuse", pointLights[i].getDiffuse());
            shader->setVec3("pointLights[i].specular", pointLights[i].getSpecular());
            shader->setFloat("pointLights[i].constant", pointLights[i].getConstant());
            shader->setFloat("pointLights[i].linear", pointLights[i].getLinear());
            shader->setFloat("pointLights[i].quadratic", pointLights[i].getQuadratic());
        }
    }

    
}
