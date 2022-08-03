#ifndef AREALIGHT_HPP
#define AREALIGHT_HPP

#include <PointLight.hpp>
#include <vector>

namespace Odysseus
{
    class AreaLight : public Light
    {
        private:
        std::vector<PointLight*> pointLights;

        public:
        AreaLight();
        virtual void start();
        virtual void update();

        virtual void setOrderOfExecution(const short& newOrderOfExecution);

        virtual short getUniqueID();

        virtual std::string toString();
        void addLight(PointLight* pt);
        void setLightShader(Odysseus::Shader* shader) const;
    };
}

#endif