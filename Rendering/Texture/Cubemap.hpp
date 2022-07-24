#ifndef CUBEMAP_HPP
#define CUBEMAP_HPP
#include <Shader.hpp>
#include <vector>
#include <Camera.hpp>

namespace Odysseus
{
    class Cubemap
    {
        public:
            unsigned int cubemapVAO, cubemapVBO;
            unsigned int cubemapTexture;
            Shader cubemapShader;
            Cubemap();

            unsigned int loadCubemap(std::vector<std::string>& faces);
            void setupCubemap();

            void update();
    };
}

#endif