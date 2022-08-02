#include "../PhongMaterial.hpp"

namespace Odysseus
{
    PhongMaterial::PhongMaterial(): Diffuse(Athena::Vector3(0.0f, 0.0f, 0.0f)), Specular(Athena::Vector3(0.0f, 0.0f, 0.0f)),
                        Ambient(Athena::Vector3(0.0f, 0.0f, 0.0f)), Shininess(0.0f){}

    PhongMaterial::PhongMaterial(Athena::Vector3& diffuse, Athena::Vector3& specular, Athena::Vector3& ambient, float shininess) :
    Diffuse(diffuse), Specular(specular), Ambient(ambient), Shininess(shininess) {}

    PhongMaterial::PhongMaterial(std::vector<Texture2D>& textures) : Textures(textures)
    {}

    void PhongMaterial::loadShaderMaterial(Odysseus::Shader* shader)
    {
        shader->setVec3("material.diffuse", this->Diffuse);
        shader->setVec3("material.specular", this->Specular);
        shader->setVec3("material.ambient", this->Diffuse);
        shader->setFloat("material.shininess", this->Shininess);
        shader->setBool("hasTexture", false);
    }

    void PhongMaterial::loadShaderTexture(Odysseus::Shader* shader)
    {
        //shader->setFloat("material.shininess", this->Shininess);
        for(GLuint i = 0; i < this->Textures.size(); ++i)
            {
                //activate texture
                glActiveTexture(GL_TEXTURE0 + i);

                //retreive texture infos
                std::string name;
                switch(this->Textures[i].type)
                {
                    case aiTextureType_DIFFUSE:
                        name = "material.diffuseTex";
                        break;
                    case aiTextureType_SPECULAR:
                        name = "material.specularTex";
                        break;
                    case aiTextureType_AMBIENT:
                        name = "material.ambientTex";
                        break;
                    case aiTextureType_NORMALS:
                        name = "material.normalTex";
                    default:
                        break;
                }
                //set shader uniform
                shader->setInt(name.c_str(), i);
                shader->setBool("hasTexture", true);
                //bind texture
                this->Textures[i].BindTexture();
            }
    }

}

