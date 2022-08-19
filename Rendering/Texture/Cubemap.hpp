#ifndef CUBEMAP_HPP
#define CUBEMAP_HPP
#include <Shader.hpp>
#include <vector>

#include <Window.hpp>
#include <Matrix4.hpp>
#include <Quaternion.hpp>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

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
            Shader* PBRshader;

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

            Cubemap();
            //unsigned int loadCubemap(std::vector<std::string>& faces);
            //void setupCubemap();
            void setupHDRImap();
            
            void setPBRshader(Shader* shader);

            void update();
    };
}

#endif