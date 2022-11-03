#include "../PhysicsMaterial.hpp"

namespace Odysseus
{
    PhysicsMaterial::PhysicsMaterial(): albedo(Athena::Vector4(0.0f, 0.0f, 0.0f, 0.0f)), metallic(0.0f), roughness(1.0f), AO(1.0f)
    {
    }

    PhysicsMaterial::PhysicsMaterial(Athena::Vector4& albedo, float metallic, float roughness):
        albedo(albedo), metallic(metallic), roughness(roughness)
    {

    }

    PhysicsMaterial::PhysicsMaterial(std::vector<Texture2D>& PBR_textures) : PBR_textures(PBR_textures)
    {

    }

    void PhysicsMaterial::loadShaderTexture(Odysseus::Shader* textureShader)
    {
        textureShader->setVec4("material.albedoColor", this->albedo);
        textureShader->setFloat("material.metallicColor", this->metallic);
        textureShader->setFloat("material.roughnessColor", this->roughness);
        textureShader->setFloat("material.AOColor", this->AO);

        textureShader->setBool("material.hasAlbedoTexture", false);
        textureShader->setBool("material.hasMetallicTexture", false);
        textureShader->setBool("material.hasRoughnessTexture", false);
        textureShader->setBool("material.hasAOTexture", false);
        textureShader->setBool("material.hasNormalMap", false);
        
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
                    textureShader->setBool("material.hasAlbedoTexture", true);
                    name = "material.albedoMap";
                    break;
                case aiTextureType_METALNESS:
                    textureShader->setBool("material.hasMetallicTexture", true);
                    name = "material.metallicMap";
                    break;
                case aiTextureType_DIFFUSE_ROUGHNESS:
                    textureShader->setBool("material.hasRoughnessTexture", true);
                    this->roughness = 0.0f;
                    name = "material.roughnessMap";
                    break;
                case aiTextureType_NORMALS:
                    textureShader->setBool("material.hasNormalMap", true);
                    name = "material.normalMap";
                    break;
                case aiTextureType_AMBIENT_OCCLUSION:
                    textureShader->setBool("material.hasAOTexture", true);
                    name = "material.AOMap";
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