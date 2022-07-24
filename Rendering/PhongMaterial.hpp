#ifndef MATERIAL_HPP
#define MATERIAL_HPP
#include <coeus.hpp>
#include "Texture/Texture2D.hpp"
#include "Shader.hpp"
#include <vector>

namespace Odysseus
{
    class Material
    {
    public:
        Athena::Vector3 Diffuse;
        Athena::Vector3 Specular;
        Athena::Vector3 Ambient;
        float Shininess;
        std::vector<Texture2D> Textures;

        Material();
        Material(Athena::Vector3& diffuse, Athena::Vector3& specular, Athena::Vector3& ambient, float shininess);
        Material(std::vector<Texture2D>& textures);

        void loadShaderMaterial(Odysseus::Shader* shader);
        void loadShaderTexture(Odysseus::Shader* shader);
    };
}

#endif