#ifndef LIGHT_HPP
#define LIGHT_HPP

#include <Vector3.hpp>
#include <Shader.hpp>
#include <Camera.hpp>

namespace System {
    class Component;
}

namespace Odysseus
{
    class System::Component;

    class Light : public System::Component
    {
        protected:
            Athena::Vector3 _diffuse;
            Athena::Vector3 _ambient;
            Athena::Vector3 _specular;
            Shader* shader;

        public:
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