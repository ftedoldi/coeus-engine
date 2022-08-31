#ifndef LIGHT_HPP
#define LIGHT_HPP

#include <Component.hpp>

#include <SerializableClass.hpp>

#include <Vector3.hpp>
#include <Shader.hpp>

#include <UUID.hpp>

namespace Odysseus
{
    class Light : public System::Component
    {
        public:
            Athena::Vector3 _diffuse;
            Athena::Vector3 _ambient;
            Athena::Vector3 _specular;

            System::UUID ID;

            Shader* shader;

            Athena::Vector3 getDiffuse() const;
            Athena::Vector3 getAmbient() const;
            Athena::Vector3 getSpecular() const;
            
            void setPosition(Athena::Vector3& pos);
            void setDiffuse(Athena::Vector3& diff);
            void setAmbient(Athena::Vector3& amb);
            void setSpecular(Athena::Vector3& spec);
            void setShader(Shader* shader);

            bool operator == (const Light& light);

            virtual void setLightShader(Odysseus::Shader* shader) const = 0;
            virtual void setLightShader(Odysseus::Shader* shader, int index) const {}
    };
}

#endif