#ifndef LIGHT_HPP
#define LIGHT_HPP

#include <Component.hpp>

#include <Vector3.hpp>
#include <Shader.hpp>

namespace Odysseus
{
    class Light : public System::Component
    {
        public:
            Athena::Vector3 _diffuse;
            Athena::Vector3 _ambient;
            Athena::Vector3 _specular;
            Shader* shader;

            Athena::Vector3 getDiffuse() const;
            Athena::Vector3 getAmbient() const;
            Athena::Vector3 getSpecular() const;
            
            void setPosition(Athena::Vector3& pos);
            void setDiffuse(Athena::Vector3& diff);
            void setAmbient(Athena::Vector3& amb);
            void setSpecular(Athena::Vector3& spec);
            void setShader(Shader* shader);

            virtual void setLightShader(Odysseus::Shader* shader) const = 0;
    };
}

#endif