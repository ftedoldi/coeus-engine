#include "../PhongMaterial.hpp"

namespace Odysseus
{
    PhongMaterial::PhongMaterial(): Diffuse(Athena::Vector3(0.8f, 0.8f, 0.8f)), Specular(Athena::Vector3(0.1f, 0.1f, 0.1f)),
                        Ambient(Athena::Vector3(0.2f, 0.2f, 0.2f)), Shininess(32.0f){}

    void PhongMaterial::loadShaderTexture(Odysseus::Shader* shader)
    {
        shader->setVec3("material.diffuse", this->Diffuse);
        shader->setVec3("material.specular", this->Specular);
        shader->setFloat("material.shininess", this->Shininess);
        //this->Diffuse.print();

        shader->setBool("material.hasDiffuseTexture", false);
        shader->setBool("material.hasSpecularTexture", false);
        shader->setBool("material.hasNormalTexture", false);

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
                        shader->setBool("material.hasDiffuseTexture", true);
                        break;
                    case aiTextureType_SPECULAR:
                        name = "material.specularTex";
                        shader->setBool("material.hasSpecularTexture", true);
                        break;
                    case aiTextureType_NORMALS:
                        name = "material.normalTex";
                        shader->setBool("material.hasNormalTexture", true);
                        break;
                    default:
                        break;
                }
                //set shader uniform
                shader->setInt(name.c_str(), i);
                //bind texture
                this->Textures[i].BindTexture();
            }
    }

}

