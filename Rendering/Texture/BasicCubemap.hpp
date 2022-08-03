#ifndef BASICCUBEMAP_HPP
#define BASICCUBEMAP_HPP
#include <Shader.hpp>
#include <vector>
#include <Camera.hpp>

namespace Odysseus
{
    class BasicCubemap
    {
        private:
            unsigned int loadCubemap(std::vector<std::string>& faces);
            void setupCubemap();
        public:
            unsigned int cubemapVAO, cubemapVBO;
            unsigned int cubemapTexture;
            Shader cubemapShader;
            BasicCubemap();

            void update();
    };
}

#endif