#include "../PhysicsMaterial.hpp"

namespace Odysseus
{
    PhysicsMaterial::PhysicsMaterial(): albedo(Athena::Vector3(0.0f, 0.0f, 0.0f)), metallic(0.0f), roughness(0.0f), AO(0.0f)
    {

    }

    PhysicsMaterial::PhysicsMaterial(Athena::Vector3& albedo, float metallic, float roughness):
        albedo(albedo), metallic(metallic), roughness(roughness)
    {

    }

    PhysicsMaterial::PhysicsMaterial(std::vector<Texture2D>& PBR_textures) : PBR_textures(PBR_textures)
    {

    }

    void PhysicsMaterial::loadShaderMaterial(Odysseus::Shader* shader)
    {

    }
    void PhysicsMaterial::loadShaderTexture(Odysseus::Shader* shader)
    {
        for(GLuint i = 0; i < this->PBR_textures.size(); ++i)
            {
                //activate texture
                //we start from GL_TEXTURE3 because:
                //GL_TEXTURE0 is reserved to irradiance map
                //GL_TEXTURE1 is reserved to prefilter map
                //GL_TEXTURE2 is reserved to brdfLUT map
                glActiveTexture(GL_TEXTURE3 + i);

                //retreive texture infos
                std::string name;
                switch(this->PBR_textures[i].type)
                {
                    case aiTextureType_DIFFUSE:
                        name = "albedoMap";
                        break;
                    case aiTextureType_METALNESS:
                        name = "metallicMap";
                        break;
                    case aiTextureType_DIFFUSE_ROUGHNESS:
                        name = "roughnessMap";
                        break;
                    case aiTextureType_NORMALS:
                        name = "normalMap";
                        break;
                    default:
                        break;
                }
                //set shader uniform
                shader->setInt(name.c_str(), i + 3);
                //shader->setBool("hasTexture", true);
                //bind texture
                this->PBR_textures[i].BindTexture();
            }
    }
}