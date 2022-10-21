#ifndef CUBEMAP_HPP
#define CUBEMAP_HPP

#include <Window.hpp>
#include <Shader.hpp>
#include <Matrix4.hpp>
#include <Quaternion.hpp>

#include <vector>

namespace Odysseus
{
    class EditorCamera;
    
    class Cubemap
    {
        private:
            unsigned int cubemapVAO, cubemapVBO;
            unsigned int captureFBO, captureRBO;
            unsigned int quadVAO;
            unsigned int quadVBO;
            unsigned int envCubemap;
            unsigned int hdrTexture;
            unsigned int irradianceMap;
            unsigned int prefilterMap;
            unsigned int brdfLUTtexture;
            unsigned int cubemapTexture;

            Shader cubemapShader;
            Shader equirectangularToCubemapShader;
            Shader irradianceShader;
            Shader prefilterShader;
            Shader brdfShader;

            void generateCube();
            void generateQuad();
            void setupShaders();
            void createSetupBuffers();
            void loadHDRImap();
            void createEnviromentMap();
            void convertToCubemap(Athena::Matrix4& captureProjection, Athena::Quaternion* captureQuatViews);
            void createIrradianceMap(Athena::Matrix4& captureProjection, Athena::Quaternion* captureQuatViews);
            void createPrefilterMap(Athena::Matrix4& captureProjection, Athena::Quaternion* captureQuatViews);

            void createLUTtexture();

        public:
            Shader* PBRshader;

            static Cubemap* currentCubemap;

            Cubemap();
            //unsigned int loadCubemap(std::vector<std::string>& faces);
            //void setupCubemap();
            void setupHDRImap();
            
            void setPBRshader(Shader* shader);

            void update();
    };
}

#endif