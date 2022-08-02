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
            Athena::Vector3 albedo;
            float metallic;
            float roughness;
            float AO;
            std::vector<Texture2D> PBR_textures;

        PhysicsMaterial();
        PhysicsMaterial(Athena::Vector3& color, float metallic, float roughness);
        PhysicsMaterial(std::vector<Texture2D>& PBR_textures);

        void loadShaderMaterial(Odysseus::Shader* shader);
        void loadShaderTexture(Odysseus::Shader* shader);
    };
} // namespace Odysseus


#endif // __PHYSICSMATERIAL_H__