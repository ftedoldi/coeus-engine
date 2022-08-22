#include "../AreaLight.hpp"

namespace Odysseus
{
    AreaLight::AreaLight()
    {
        _ambient = Athena::Vector3();
        _diffuse = Athena::Vector3(0.5f, 0.5f, 0.5f);
        _specular = Athena::Vector3();

        auto pLight = new PointLight();

        pointLights.push_back(pLight);
        
        shader = new Odysseus::Shader(".\\Shader\\phongShader.vert", ".\\Shader\\phongShader.frag");
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

    int AreaLight::getUniqueID()
    {
        return this->_uniqueID;
    }

    std::string AreaLight::toString()
    {
        return "AreaLight";
    }

    void AreaLight::addLight(PointLight* pt)
    {
        this->pointLights.push_back(pt);
    }

    void AreaLight::setLightShader(Odysseus::Shader* shader) const
    {
        this->shader->use();
        
        for(unsigned int i = 0; i < this->pointLights.size(); ++i)
        {
            std::cout << "pointlight" << i << " position: ";
            pointLights[i]->getPosition().print();
            std::cout << std::endl;
            shader->setVec3("pointLights[i].position", pointLights[i]->getPosition());
            shader->setVec3("pointLights[i].ambient", pointLights[i]->getAmbient());
            shader->setVec3("pointLights[i].diffuse", pointLights[i]->getDiffuse());
            shader->setVec3("pointLights[i].specular", pointLights[i]->getSpecular());
            shader->setFloat("pointLights[i].constant", pointLights[i]->getConstant());
            shader->setFloat("pointLights[i].linear", pointLights[i]->getLinear());
            shader->setFloat("pointLights[i].quadratic", pointLights[i]->getQuadratic());
        }
    }

    
}
