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

            virtual void showComponentFieldsInEditor();

            virtual void serialize(YAML::Emitter& out);
            virtual System::Component* deserialize(YAML::Node& node);
            
            void addLight(PointLight* pt);
            void setLightShader(Odysseus::Shader* shader) const;
            void setLightShader(Odysseus::Shader* shader, int index) const;

            ~AreaLight();
                                    
            SERIALIZABLE_CLASS(System::Component, Light);
    };
}

#endif