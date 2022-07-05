#ifndef LIGHT_HPP
#define LIGHT_HPP
#include <Vector3.hpp>
#include <Shader.hpp>

namespace Odysseus
{
    class Light
    {
        protected:
            Athena::Vector3 _diffuse;
            Athena::Vector3 _ambient;
            Athena::Vector3 _specular;

        public:
            Light(Athena::Vector3& diffuse, Athena::Vector3& specular, Athena::Vector3& ambient);
            Athena::Vector3 getDiffuse() const;
            Athena::Vector3 getAmbient() const;
            Athena::Vector3 getSpecular() const;
            
            void setPosition(Athena::Vector3& pos);
            void setDiffuse(Athena::Vector3& diff);
            void setAmbient(Athena::Vector3& amb);
            void setSpecular(Athena::Vector3& spec);

            virtual void setLightShader(Odysseus::Shader& shader) const = 0;

    };
}

#endif