#ifndef AREALIGHT_HPP
#define AREALIGHT_HPP

#include <PointLight.hpp>
#include <vector>

namespace Odysseus
{
    class AreaLight : public Light
    {
        public:
            std::vector<PointLight*> pointLights;
            AreaLight();
            virtual void start();
            virtual void update();
    
            virtual void setOrderOfExecution(const short& newOrderOfExecution);
    
            virtual int getUniqueID();
    
            virtual std::string toString();
            void addLight(PointLight* pt);
            void setLightShader(Odysseus::Shader* shader) const;
    };
}

#endif