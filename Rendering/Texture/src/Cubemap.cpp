#include "../Cubemap.hpp"

#include <stb/stb_image.h>

#include <EditorCamera.hpp>

namespace Odysseus
{
    Cubemap::Cubemap()
    {
        //this->setupCubemap();
        setupHDRImap();
    }

    /*void Cubemap::setupCubemap()
    {
        stbi_set_flip_vertically_on_load(false);
        generateCube();

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
        
    }*/

    void Cubemap::setupShaders()
    {
        this->cubemapShader.assignShadersPath(".\\Shader\\cubemapShader.vert", ".\\Shader\\cubemapShader.frag");
        this->equirectangularToCubemapShader.assignShadersPath(".\\Shader\\HDRImapShader.vert", ".\\Shader\\HDRImapShader.frag");
        this->irradianceShader.assignShadersPath(".\\Shader\\HDRImapShader.vert", ".\\Shader\\irradianceShader.frag");
        this->prefilterShader.assignShadersPath(".\\Shader\\HDRImapShader.vert", ".\\Shader\\prefilterShader.frag");
        this->brdfShader.assignShadersPath(".\\Shader\\brdfShader.vert", ".\\Shader\\brdfShader.frag");
        cubemapShader.use();
        cubemapShader.setInt("environmentMap", 0);
    }

    void Cubemap::createSetupBuffers()
    {
        //create frame buffer object and render buffer object
        glGenFramebuffers(1, &this->captureFBO);
        glGenRenderbuffers(1, &this->captureRBO);

        glBindFramebuffer(GL_FRAMEBUFFER, this->captureFBO);
        glBindRenderbuffer(GL_RENDERBUFFER, this->captureRBO);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, 512, 512);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, this->captureRBO);
    }

    void Cubemap::loadHDRImap()
    {
        int width, height, nrComponents;
        //float *data = stbi_loadf(".\\Assets\\Models\\HDRImap\\PaperMill_A_3k.hdr", &width, &height, &nrComponents, 0);
        float *data = stbi_loadf(".\\Assets\\Models\\HDRImap\\Newport_Loft_Ref.hdr", &width, &height, &nrComponents, 0);
        if (data)
        {
            glGenTextures(1, &this->hdrTexture);
            glBindTexture(GL_TEXTURE_2D, this->hdrTexture);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, width, height, 0, GL_RGB, GL_FLOAT, data); 

            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

            stbi_image_free(data);
        }
        else
        {
            std::cout << "Failed to load HDR image." << std::endl;
        }
    }

    void Cubemap::createEnviromentMap()
    {
        glGenTextures(1, &this->envCubemap);
        glBindTexture(GL_TEXTURE_CUBE_MAP, this->envCubemap);
        for (unsigned int i = 0; i < 6; ++i)
        {
            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB16F, 512, 512, 0, GL_RGB, GL_FLOAT, nullptr);
        }
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
        //set GL_TEXTURE_MIN_FILTER to GL_LINEAR_MIPMAP_LINEAR to make mipmap sampling
        //during pre-filter convolution, work
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR); 
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    }

    void Cubemap::convertToCubemap(Athena::Matrix4& captureProjection, Athena::Quaternion* captureQuatViews)
    {
        //convert the HDR equirectancular enviroment map created before, to the cubemap equivalent
        //to do this we need to render an unit cube and project the equirectangular map on each
        //of the cube faces (6 in total)
        //the vertex shader simply renders the cube and pass its local positions to the fragment shader
        this->equirectangularToCubemapShader.use();
        this->equirectangularToCubemapShader.setInt("equirectangularMap", 0);
        this->equirectangularToCubemapShader.setMat4("projection", captureProjection);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, this->hdrTexture);

        glViewport(0, 0, 512, 512);
        glBindFramebuffer(GL_FRAMEBUFFER, this->captureFBO);
        for(unsigned int i = 0; i < 6; ++i)
        {
            Athena::Quaternion rotation = captureQuatViews[i].inverse();
            this->equirectangularToCubemapShader.setVec4("rotation", rotation.asVector4());
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, this->envCubemap, 0);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            generateCube();
        }
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        // then let OpenGL generate mipmaps from first mip face (combatting visible dots artifact)
        glBindTexture(GL_TEXTURE_CUBE_MAP, this->envCubemap);
        glGenerateMipmap(GL_TEXTURE_CUBE_MAP);
    }

    void Cubemap::createIrradianceMap(Athena::Matrix4& captureProjection, Athena::Quaternion* captureQuatViews)
    {
        //the irradiance map represents the diffuse part of the reflectance integral
        //it holds all of the scene's indirect diffuse light
        glGenTextures(1, &this->irradianceMap);
        glBindTexture(GL_TEXTURE_CUBE_MAP, this->irradianceMap);
        for (unsigned int i = 0; i < 6; ++i)
        {
            //since the map doesnt need a lot of high frequency details, we can store it at low resolution
            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB16F, 32, 32, 0, GL_RGB, GL_FLOAT, nullptr);
        }
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        glBindFramebuffer(GL_FRAMEBUFFER, this->captureFBO);
        glBindRenderbuffer(GL_RENDERBUFFER, this->captureRBO);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, 32, 32);

        this->irradianceShader.use();
        this->irradianceShader.setInt("environmentMap", 0);
        this->irradianceShader.setMat4("projection", captureProjection);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_CUBE_MAP, this->envCubemap);

        //setting the viewport as the resolution of the map
        glViewport(0, 0, 32, 32);
        glBindFramebuffer(GL_FRAMEBUFFER, this->captureFBO);
        for (unsigned int i = 0; i < 6; ++i)
        {
            Athena::Quaternion rotation = captureQuatViews[i].inverse();
            this->irradianceShader.setVec4("rotation", rotation.asVector4());
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, this->irradianceMap, 0);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            generateCube();
        }
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    void Cubemap::createPrefilterMap(Athena::Matrix4& captureProjection, Athena::Quaternion* captureQuatViews)
    {
        // pbr: create a pre-filter cubemap, and re-scale capture FBO to pre-filter scale.
        // --------------------------------------------------------------------------------
        //pre-filter cubemap, similar to the irradiance map but by taking into account the roughness.
        //To increase the roughness 
        glGenTextures(1, &this->prefilterMap);
        glBindTexture(GL_TEXTURE_CUBE_MAP, this->prefilterMap);
        for (unsigned int i = 0; i < 6; ++i)
        {
            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB16F, 512, 512, 0, GL_RGB, GL_FLOAT, nullptr);
        }
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR); // be sure to set minification filter to mip_linear 
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        // generate mipmaps for the cubemap so OpenGL automatically allocates the required memory.
        glGenerateMipmap(GL_TEXTURE_CUBE_MAP);

        // pbr: run a quasi monte-carlo simulation on the environment lighting to create a prefilter (cube)map.
        this->prefilterShader.use();
        this->prefilterShader.setInt("environmentMap", 0);
        this->prefilterShader.setMat4("projection", captureProjection);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_CUBE_MAP, this->envCubemap);

        glBindFramebuffer(GL_FRAMEBUFFER, this->captureFBO);
        unsigned int maxMipLevels = 5;
        for (unsigned int mip = 0; mip < maxMipLevels; ++mip)
        {
            // resize framebuffer according to mip-level size.
            unsigned int mipWidth = static_cast<unsigned int>(512 * std::pow(0.5, mip));
            unsigned int mipHeight = static_cast<unsigned int>(512 * std::pow(0.5, mip));
            glBindRenderbuffer(GL_RENDERBUFFER, this->captureRBO);
            glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, mipWidth, mipHeight);
            glViewport(0, 0, mipWidth, mipHeight);

            float roughness = (float)mip / ((float)(maxMipLevels - 1) + 0.00001);
            this->prefilterShader.setFloat("roughness", roughness);
            for (unsigned int i = 0; i < 6; ++i)
            {
                Athena::Quaternion rotation = captureQuatViews[i].inverse();
                this->prefilterShader.setVec4("rotation", rotation.asVector4());
                glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, this->prefilterMap, mip);

                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
                generateCube();
            }
        }
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    void Cubemap::createLUTtexture()
    {
        // pbr: generate a 2D LUT from the BRDF equations used.
        // ----------------------------------------------------
        glGenTextures(1, &this->brdfLUTtexture);

        // pre-allocate enough memory for the LUT texture.
        glBindTexture(GL_TEXTURE_2D, this->brdfLUTtexture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RG16F, 512, 512, 0, GL_RG, GL_FLOAT, 0);
        // be sure to set wrapping mode to GL_CLAMP_TO_EDGE
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        // then re-configure capture framebuffer object and render screen-space quad with BRDF shader.
        glBindFramebuffer(GL_FRAMEBUFFER, this->captureFBO);
        glBindRenderbuffer(GL_RENDERBUFFER, this->captureRBO);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, 512, 512);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, this->brdfLUTtexture, 0);

        glViewport(0, 0, 512, 512);
        this->brdfShader.use();
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        generateQuad();

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    void Cubemap::setupHDRImap()
    {
        glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);
        glDepthFunc(GL_LEQUAL);
        setupShaders();
        createSetupBuffers();
        loadHDRImap();
        createEnviromentMap();

        //set up projection and view matrices for capturing data onto the 6 cubemap face directions
        //each quaternion looks to one of the 6 possible faces
        //the projection matrix with a FOV of 90, captures the entire face
        //What we'll do after is render a cube 6 times, storing the result in a floating point framebuffer
        Athena::Matrix4 captureProjection = Odysseus::EditorCamera::perspective(90.0f, 1.0f, 0.1f, 10.0f);
        Athena::Quaternion captureQuatViews[] = 
        {
            Athena::Quaternion::matToQuatCast(Odysseus::EditorCamera::lookAt(Athena::Vector3(0.0f, 0.0f, 0.0f), Athena::Vector3(1.0f, 0.0f, 0.0f), Athena::Vector3(0.0f, -1.0f, 0.0f))),
            Athena::Quaternion::matToQuatCast(Odysseus::EditorCamera::lookAt(Athena::Vector3(0.0f, 0.0f, 0.0f), Athena::Vector3(-1.0f, 0.0f, 0.0f), Athena::Vector3(0.0f, -1.0f, 0.0f))),
            Athena::Quaternion::matToQuatCast(Odysseus::EditorCamera::lookAt(Athena::Vector3(0.0f, 0.0f, 0.0f), Athena::Vector3(0.0f, 1.0f, 0.0f), Athena::Vector3(0.0f, 0.0f, 1.0f))),
            Athena::Quaternion::matToQuatCast(Odysseus::EditorCamera::lookAt(Athena::Vector3(0.0f, 0.0f, 0.0f), Athena::Vector3(0.0f, -1.0f, 0.0f), Athena::Vector3(0.0f, 0.0f, -1.0f))),
            Athena::Quaternion::matToQuatCast(Odysseus::EditorCamera::lookAt(Athena::Vector3(0.0f, 0.0f, 0.0f), Athena::Vector3(0.0f, 0.0f, 1.0f), Athena::Vector3(0.0f, -1.0f, 0.0f))),
            Athena::Quaternion::matToQuatCast(Odysseus::EditorCamera::lookAt(Athena::Vector3(0.0f, 0.0f, 0.0f), Athena::Vector3(0.0f, 0.0f, -1.0f), Athena::Vector3(0.0f, -1.0f, 0.0f)))
        };

        convertToCubemap(captureProjection, captureQuatViews);
        createIrradianceMap(captureProjection, captureQuatViews);
        createPrefilterMap(captureProjection, captureQuatViews);
        createLUTtexture();

        Athena::Matrix4 projection = Odysseus::SceneManager::activeScene->sceneEditor->editorCamera->perspective(45.0f, System::Window::screen.width / System::Window::screen.height, 0.1f, 100.0f);
        cubemapShader.use();
        cubemapShader.setMat4("projection", projection);

        int SRCwidth, SRCheight;
        glfwGetFramebufferSize(System::Window::window, &SRCwidth, &SRCheight);
        glViewport(0, 0, SRCwidth, SRCheight);
    }

    /*unsigned int Cubemap::loadCubemap(std::vector<std::string>& faces)
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
    }*/

    void Cubemap::update()
    {
        glDepthFunc(GL_LEQUAL);
        glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);
        this->PBRshader->use();
        
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_CUBE_MAP, this->irradianceMap);

        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_CUBE_MAP, this->prefilterMap);

        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_2D, this->brdfLUTtexture);

        cubemapShader.use();
        
        cubemapShader.setVec4("rotation", Odysseus::SceneManager::activeScene->sceneEditor->editorCamera->transform->rotation.inverse().asVector4());
        //cubemapShader.setMat4("projection", Odysseus::SceneManager::activeScene->sceneEditor->editorCamera->perspective(45.0f, System::Window::screen.width / System::Window::screen.height, 0.1f, 100.0f));
        cubemapShader.setInt("environmentMap", 0);

        // skybox cube
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_CUBE_MAP, this->envCubemap);
        //glBindTexture(GL_TEXTURE_CUBE_MAP, this->prefilterMap);
        //glBindTexture(GL_TEXTURE_CUBE_MAP, this->irradianceMap);
        generateCube();
        glDepthFunc(GL_LESS); // set depth function back to default
        glDisable(GL_TEXTURE_CUBE_MAP_SEAMLESS);

        //brdfShader.use();
        //generateQuad();
    }

    void Cubemap::generateCube()
    {
        float vertices[] = {
            // back face
            -1.0f, -1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 0.0f, 0.0f, // bottom-left
            1.0f,  1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 1.0f, 1.0f, // top-right
            1.0f, -1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 1.0f, 0.0f, // bottom-right         
            1.0f,  1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 1.0f, 1.0f, // top-right
            -1.0f, -1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 0.0f, 0.0f, // bottom-left
            -1.0f,  1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 0.0f, 1.0f, // top-left
            // front face
            -1.0f, -1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 0.0f, 0.0f, // bottom-left
            1.0f, -1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 1.0f, 0.0f, // bottom-right
            1.0f,  1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 1.0f, 1.0f, // top-right
            1.0f,  1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 1.0f, 1.0f, // top-right
            -1.0f,  1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 0.0f, 1.0f, // top-left
            -1.0f, -1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 0.0f, 0.0f, // bottom-left
            // left face
            -1.0f,  1.0f,  1.0f, -1.0f,  0.0f,  0.0f, 1.0f, 0.0f, // top-right
            -1.0f,  1.0f, -1.0f, -1.0f,  0.0f,  0.0f, 1.0f, 1.0f, // top-left
            -1.0f, -1.0f, -1.0f, -1.0f,  0.0f,  0.0f, 0.0f, 1.0f, // bottom-left
            -1.0f, -1.0f, -1.0f, -1.0f,  0.0f,  0.0f, 0.0f, 1.0f, // bottom-left
            -1.0f, -1.0f,  1.0f, -1.0f,  0.0f,  0.0f, 0.0f, 0.0f, // bottom-right
            -1.0f,  1.0f,  1.0f, -1.0f,  0.0f,  0.0f, 1.0f, 0.0f, // top-right
            // right face
            1.0f,  1.0f,  1.0f,  1.0f,  0.0f,  0.0f, 1.0f, 0.0f, // top-left
            1.0f, -1.0f, -1.0f,  1.0f,  0.0f,  0.0f, 0.0f, 1.0f, // bottom-right
            1.0f,  1.0f, -1.0f,  1.0f,  0.0f,  0.0f, 1.0f, 1.0f, // top-right         
            1.0f, -1.0f, -1.0f,  1.0f,  0.0f,  0.0f, 0.0f, 1.0f, // bottom-right
            1.0f,  1.0f,  1.0f,  1.0f,  0.0f,  0.0f, 1.0f, 0.0f, // top-left
            1.0f, -1.0f,  1.0f,  1.0f,  0.0f,  0.0f, 0.0f, 0.0f, // bottom-left     
            // bottom face
            -1.0f, -1.0f, -1.0f,  0.0f, -1.0f,  0.0f, 0.0f, 1.0f, // top-right
            1.0f, -1.0f, -1.0f,  0.0f, -1.0f,  0.0f, 1.0f, 1.0f, // top-left
            1.0f, -1.0f,  1.0f,  0.0f, -1.0f,  0.0f, 1.0f, 0.0f, // bottom-left
            1.0f, -1.0f,  1.0f,  0.0f, -1.0f,  0.0f, 1.0f, 0.0f, // bottom-left
            -1.0f, -1.0f,  1.0f,  0.0f, -1.0f,  0.0f, 0.0f, 0.0f, // bottom-right
            -1.0f, -1.0f, -1.0f,  0.0f, -1.0f,  0.0f, 0.0f, 1.0f, // top-right
            // top face
            -1.0f,  1.0f, -1.0f,  0.0f,  1.0f,  0.0f, 0.0f, 1.0f, // top-left
            1.0f,  1.0f , 1.0f,  0.0f,  1.0f,  0.0f, 1.0f, 0.0f, // bottom-right
            1.0f,  1.0f, -1.0f,  0.0f,  1.0f,  0.0f, 1.0f, 1.0f, // top-right     
            1.0f,  1.0f,  1.0f,  0.0f,  1.0f,  0.0f, 1.0f, 0.0f, // bottom-right
            -1.0f,  1.0f, -1.0f,  0.0f,  1.0f,  0.0f, 0.0f, 1.0f, // top-left
            -1.0f,  1.0f,  1.0f,  0.0f,  1.0f,  0.0f, 0.0f, 0.0f  // bottom-left        
        };
        glGenVertexArrays(1, &this->cubemapVAO);
        glGenBuffers(1, &this->cubemapVBO);
        // fill buffer
        glBindBuffer(GL_ARRAY_BUFFER, this->cubemapVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
        // link vertex attributes
        glBindVertexArray(this->cubemapVAO);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);
        // render Cube
        glBindVertexArray(this->cubemapVAO);
        glDrawArrays(GL_TRIANGLES, 0, 36);
        glBindVertexArray(0);
    }

    void Cubemap::setPBRshader(Shader* shader)
    {
        this->PBRshader = shader;
        this->PBRshader->use();
        this->PBRshader->setInt("irradianceMap", 0);
        this->PBRshader->setInt("prefilterMap", 1);
        this->PBRshader->setInt("brdfLUT", 2);
    }

    void Cubemap::generateQuad()
    {
        float quadVertices[] = {
            // positions        // texture Coords
            -1.0f,  1.0f, 0.0f, 0.0f, 1.0f,
            -1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
            1.0f,  1.0f, 0.0f, 1.0f, 1.0f,
            1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
        };
        // setup plane VAO
        glGenVertexArrays(1, &this->quadVAO);
        glGenBuffers(1, &this->quadVBO);
        glBindVertexArray(this->quadVAO);
        glBindBuffer(GL_ARRAY_BUFFER, this->quadVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
        glBindVertexArray(this->quadVAO);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        glBindVertexArray(0);
    }
    
}

