#include "../Material.hpp"

namespace Odysseus
{
    Material::Material(): Diffuse(Athena::Vector4(0.0f, 0.0f, 0.0f, 0.0f)), Specular(Athena::Vector4(0.0f, 0.0f, 0.0f, 0.0f)),
                        Ambient(Athena::Vector4(0.0f, 0.0f, 0.0f, 0.0f)), Shininess(0.0f){}

    Material::Material(Athena::Vector4& diffuse, Athena::Vector4& specular, Athena::Vector4& ambient, float shininess) :
    Diffuse(diffuse), Specular(specular), Ambient(ambient), Shininess(shininess) {}

    Material::Material(std::vector<Texture2D>& textures) : Textures(textures)
    {}

    void Material::loadShaderMaterial(Odysseus::Shader& shader)
    {
        shader.setVec4("material.diffuse", this->Diffuse.coordinates.x, this->Diffuse.coordinates.y, this->Diffuse.coordinates.z, 1.0f);
        shader.setVec4("material.specular", this->Specular.coordinates.x, this->Specular.coordinates.y, this->Specular.coordinates.z, 1.0f);
        shader.setVec4("material.ambient", this->Ambient.coordinates.x, this->Ambient.coordinates.y, this->Ambient.coordinates.z, 1.0f);
        shader.setFloat("material.shininess", this->Shininess);
        shader.setBool("hasTexture", false);
    }

    void Material::loadShaderTexture(Odysseus::Shader& shader)
    {
        GLuint diffuseIdx = 0;
        GLuint specularIdx = 0;
        GLuint heightIdx = 0;
        GLuint ambientIdx = 0;

        for(GLuint i = 0; i < this->Textures.size(); ++i)
            {
                //activate texture
                glActiveTexture(GL_TEXTURE0 + i);

                //retreive texture infos
                std::string name;
                switch(this->Textures[i].type)
                {
                    case aiTextureType_DIFFUSE:
                        name = "diffuse" + std::to_string(diffuseIdx++);
                        break;
                    case aiTextureType_SPECULAR:
                        name = "specular" + std::to_string(specularIdx++);
                        break;
                    case aiTextureType_HEIGHT:
                        name = "height" + std::to_string(heightIdx++);
                        break;
                    case aiTextureType_AMBIENT:
                        name = "ambient" + std::to_string(ambientIdx++);
                        break;
                    default:
                        break;
                }
                //set shader uniform
                shader.setInt(name.c_str(), i);
                shader.setBool("hasTexture", true);
                //bind texture
                this->Textures[i].BindTexture();
            }
    }

}

