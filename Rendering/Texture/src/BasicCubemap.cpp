#include "../BasicCubemap.hpp"

#include <Window.hpp>

#include <stb/stb_image.h>

namespace Odysseus
{
    BasicCubemap::BasicCubemap()
    {
        this->setupCubemap();
    }

    void BasicCubemap::setupCubemap()
    {
        stbi_set_flip_vertically_on_load(false);
        float cubemapVertices[] = {
            // positions          
            -1.0f,  1.0f, -1.0f,
            -1.0f, -1.0f, -1.0f,
            1.0f, -1.0f, -1.0f,
            1.0f, -1.0f, -1.0f,
            1.0f,  1.0f, -1.0f,
            -1.0f,  1.0f, -1.0f,

            -1.0f, -1.0f,  1.0f,
            -1.0f, -1.0f, -1.0f,
            -1.0f,  1.0f, -1.0f,
            -1.0f,  1.0f, -1.0f,
            -1.0f,  1.0f,  1.0f,
            -1.0f, -1.0f,  1.0f,

            1.0f, -1.0f, -1.0f,
            1.0f, -1.0f,  1.0f,
            1.0f,  1.0f,  1.0f,
            1.0f,  1.0f,  1.0f,
            1.0f,  1.0f, -1.0f,
            1.0f, -1.0f, -1.0f,

            -1.0f, -1.0f,  1.0f,
            -1.0f,  1.0f,  1.0f,
            1.0f,  1.0f,  1.0f,
            1.0f,  1.0f,  1.0f,
            1.0f, -1.0f,  1.0f,
            -1.0f, -1.0f,  1.0f,

            -1.0f,  1.0f, -1.0f,
            1.0f,  1.0f, -1.0f,
            1.0f,  1.0f,  1.0f,
            1.0f,  1.0f,  1.0f,
            -1.0f,  1.0f,  1.0f,
            -1.0f,  1.0f, -1.0f,

            -1.0f, -1.0f, -1.0f,
            -1.0f, -1.0f,  1.0f,
            1.0f, -1.0f, -1.0f,
            1.0f, -1.0f, -1.0f,
            -1.0f, -1.0f,  1.0f,
            1.0f, -1.0f,  1.0f
        };
        glGenVertexArrays(1, &this->cubemapVAO);
        glGenBuffers(1, &this->cubemapVBO);
        glBindVertexArray(this->cubemapVAO);
        glBindBuffer(GL_ARRAY_BUFFER, this->cubemapVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(cubemapVertices), &cubemapVertices, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);

        this->cubemapShader.assignShadersPath(".\\Shader\\cubemapShader.vert", ".\\Shader\\cubemapShader.frag");
        cubemapShader.use();
        cubemapShader.setInt("skybox", 0);

        std::vector<std::string> faces
        {
            ".\\Assets\\Models\\cubemap\\right.jpg",
            ".\\Assets\\Models\\cubemap\\left.jpg",
            ".\\Assets\\Models\\cubemap\\top.jpg",
            ".\\Assets\\Models\\cubemap\\bottom.jpg",
            ".\\Assets\\Models\\cubemap\\front.jpg",
            ".\\Assets\\Models\\cubemap\\back.jpg",
        };

        this->cubemapTexture = this->loadCubemap(faces);
        stbi_set_flip_vertically_on_load(true);
        
    }

    unsigned int BasicCubemap::loadCubemap(std::vector<std::string>& faces)
    {
        unsigned int textureID;
        glGenTextures(1, &textureID);
        glBindTexture(GL_TEXTURE_CUBE_MAP, textureID);

        int width, height, nrChannels;
        for (unsigned int i = 0; i < faces.size(); ++i)
        {
            unsigned char *data = stbi_load(faces[i].c_str(), &width, &height, &nrChannels, 0);
            if (data)
            {
                glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 
                            0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data
                );
                stbi_image_free(data);
            }
            else
            {
                std::cout << "Cubemap tex failed to load at path: " << faces[i] << std::endl;
                stbi_image_free(data);
            }
        }
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

        return textureID;
    }

    void BasicCubemap::update()
    {
        glDepthFunc(GL_LEQUAL);
        cubemapShader.use();
        
        cubemapShader.setVec4("rotation", Odysseus::SceneManager::activeScene->sceneEditor->editorCamera->transform->rotation.inverse().asVector4());
        cubemapShader.setMat4("projection", Odysseus::SceneManager::activeScene->sceneEditor->editorCamera->perspective(45.0f, System::Window::screen.width / System::Window::screen.height, 0.1f, 100.0f));

        // skybox cube
        glBindVertexArray(this->cubemapVAO);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_CUBE_MAP, this->cubemapTexture);
        glDrawArrays(GL_TRIANGLES, 0, 36);
        glBindVertexArray(0);
        glDepthFunc(GL_LESS); // set depth function back to default
    }
}