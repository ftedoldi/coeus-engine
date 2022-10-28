#include "../PhysicsMaterial.hpp"

namespace Odysseus
{
    PhysicsMaterial::PhysicsMaterial(): albedo(Athena::Vector4(0.0f, 0.0f, 0.0f, 0.0)), metallic(0.0f), roughness(0.0f), AO(0.0f), 
                                        hasAlbedoTexture(false), hasMetallicTexture(false), hasRoughnessTexture(false)
    {
    }

    PhysicsMaterial::PhysicsMaterial(Athena::Vector4& albedo, float metallic, float roughness):
        albedo(albedo), metallic(metallic), roughness(roughness)
    {

    }

    PhysicsMaterial::PhysicsMaterial(std::vector<Texture2D>& PBR_textures) : PBR_textures(PBR_textures)
    {

    }

    void PhysicsMaterial::loadShaderMaterial(Odysseus::Shader* materialShader)
    {
        materialShader->setVec4("material.albedoColor", this->albedo);
        materialShader->setFloat("material.metallicColor", this->metallic);
        materialShader->setFloat("material.roughnessColor", this->roughness);
    }

    void PhysicsMaterial::loadShaderTexture(Odysseus::Shader* textureShader)
    {
        textureShader->setBool("hasNormalMap", false);
        for(GLuint i = 0; i < this->PBR_textures.size(); ++i)
        {
            //activate texture
            //we start from GL_TEXTURE3 because:
            //GL_TEXTURE0 is reserved to irradiance map
            //GL_TEXTURE1 is reserved to prefilter map
            //GL_TEXTURE2 is reserved to brdfLUT map
            glActiveTexture(GL_TEXTURE4 + i);

            //retreive texture infos
            std::string name;
            switch(this->PBR_textures[i].type)
            {
                case aiTextureType_DIFFUSE:
                    name = "material.albedoMap";
                    break;
                case aiTextureType_METALNESS:
                    name = "material.metallicMap";
                    break;
                case aiTextureType_DIFFUSE_ROUGHNESS:
                    name = "material.roughnessMap";
                    break;
                case aiTextureType_NORMALS:
                    textureShader->setBool("material.hasNormalMap", true);
                    name = "material.normalMap";
                    break;
                default:
                    break;
            }
            //set shader uniform
            textureShader->setInt(name.c_str(), i + 4);
            //bind texture
            this->PBR_textures[i].BindTexture();
        }
    }
}