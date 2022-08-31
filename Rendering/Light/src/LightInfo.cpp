#include "../LightInfo.hpp"

namespace Odysseus
{
    std::vector<Odysseus::PointLight*> LightInfo::pointLights;
    std::vector<Odysseus::SpotLight*> LightInfo::spotLights;
    std::vector<Odysseus::DirectionalLight*> LightInfo::directionalLights;
    std::vector<Odysseus::AreaLight*> LightInfo::areaLights;

    std::set<Odysseus::PointLight*> LightInfo::existingPointLights;
    std::set<Odysseus::SpotLight*> LightInfo::existingSpotLights;
    std::set<Odysseus::DirectionalLight*> LightInfo::existingDirectionalLights;
    std::set<Odysseus::AreaLight*> LightInfo::existingAreaLights;

    void LightInfo::computeLighting(Shader* meshShader)
    {
        meshShader->use();
        meshShader->setInt("numberOfPointLights", pointLights.size());

        for (int i = 0; i < pointLights.size(); i++)
            pointLights[i]->setLightShader(meshShader, i);
    }

}