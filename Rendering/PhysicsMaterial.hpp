#ifndef __PHYSICSMATERIAL_H__
#define __PHYSICSMATERIAL_H__
#include <Vector3.hpp>
#include <vector>
#include <Texture2D.hpp>
#include <Shader.hpp>

namespace Odysseus
{
    class PhysicsMaterial {

        public:
            Athena::Vector4 albedo;
            float metallic;
            float roughness;
            float AO;
            std::vector<Texture2D> PBR_textures;
            bool hasAlbedoTexture;
            bool hasMetallicTexture;
            bool hasRoughnessTexture;

        PhysicsMaterial();
        PhysicsMaterial(Athena::Vector4& color, float metallic, float roughness);
        PhysicsMaterial(std::vector<Texture2D>& PBR_textures);

        void loadShaderTexture(Odysseus::Shader* shader);
    };
} // namespace Odysseus


#endif // __PHYSICSMATERIAL_H__