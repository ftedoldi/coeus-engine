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
        Athena::Vector4 Diffuse;
        Athena::Vector4 Specular;
        Athena::Vector4 Ambient;
        float Shininess;
        std::vector<Texture2D> Textures;

        Material();

        Material(Athena::Vector4& diffuse, Athena::Vector4& specular, Athena::Vector4& ambient, float shininess);

        Material(std::vector<Texture2D>& textures);

        void loadShaderMaterial(Odysseus::Shader& shader);
        void loadShaderTexture(Odysseus::Shader& shader);
    };
}

#endif