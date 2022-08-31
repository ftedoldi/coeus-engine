#ifndef __LIGHTINFO_H__
#define __LIGHTINFO_H__

#include <PointLight.hpp>
#include <SpotLight.hpp>
#include <DirectionalLight.hpp>
#include <AreaLight.hpp>

#include <UUID.hpp>

#include <Shader.hpp>

#include <vector>
#include <set>

namespace Odysseus
{
    class LightInfo
    {
        public:
            static std::vector<Odysseus::PointLight*> pointLights;
            static std::vector<Odysseus::SpotLight*> spotLights;
            static std::vector<Odysseus::DirectionalLight*> directionalLights;
            static std::vector<Odysseus::AreaLight*> areaLights;

            static std::set<Odysseus::PointLight*> existingPointLights;
            static std::set<Odysseus::SpotLight*> existingSpotLights;
            static std::set<Odysseus::DirectionalLight*> existingDirectionalLights;
            static std::set<Odysseus::AreaLight*> existingAreaLights;

            static void computeLighting(Shader* meshShader);
    };
} // namespace Odysseus


#endif // __LIGHTINFO_H__