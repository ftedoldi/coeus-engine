#include "../FrameBuffer.hpp"

#include <IO/Input.hpp>

#include <CUDAInteroperationManager.hpp>
#include <Postprocess.cuh>

namespace System::Buffers
{
    FrameBuffer::FrameBuffer() : ID(_id), 
                                 MSAA_ID(_MSAAid),
                                 frameBufferShader(_frameBufferShader),
                                 frameBufferSize(_frameBufferSize),
                                 texturesID(_texturesID),
                                 internalTextureFormat(_internalTextureFormat),
                                 textureFormat(_textureFormat),
                                 pixelValueType(_pixelValueType),
                                 postProcessedTexture(_postProcessedTexture)
    {
        _frameBufferShader = new Odysseus::Shader(".\\Shader\\frameBufferShader.vert", ".\\Shader\\frameBufferShader.frag");

        _frameBufferSize.width = 800;
        _frameBufferSize.height = 800;

        refreshColor = Athena::Vector4();

        this->initializeScreenQuad();

        _isMSAA_Buffer = true;

        this->_internalTextureFormat = GL_RGBA32F;
        this->_textureFormat = GL_RGBA;
        this->_pixelValueType = GL_FLOAT; 

        this->_numberOfColorAttachments = 2;

        this->initializeMSAABuffer();
        this->initializeStandardBuffer();
    }

    FrameBuffer::FrameBuffer(
                                const Athena::Scalar& width, 
                                const Athena::Scalar& height, 
                                bool isMSAA_Buffer,
                                int numberOfColorAttachments,
                                GLint internalTexFormat,
                                GLenum texFormat,
                                GLenum pixelValType,
                                Athena::Vector4 refreshColor
                            ) : ID(_id),
                                MSAA_ID(_MSAAid),
                                frameBufferShader(_frameBufferShader),
                                frameBufferSize(_frameBufferSize),
                                texturesID(_texturesID),
                                internalTextureFormat(_internalTextureFormat),
                                textureFormat(_textureFormat),
                                pixelValueType(_pixelValueType),
                                postProcessedTexture(_postProcessedTexture)
    {
        _frameBufferShader = new Odysseus::Shader(".\\Shader\\frameBufferShader.vert", ".\\Shader\\frameBufferShader.frag");

        _frameBufferSize.width = width;
        _frameBufferSize.height = height;

        this->refreshColor = refreshColor;

        this->initializeScreenQuad();

        this->_isMSAA_Buffer = isMSAA_Buffer;

        this->_numberOfColorAttachments = numberOfColorAttachments < 1 ? 1 : numberOfColorAttachments > 12 ? 12 : numberOfColorAttachments;

        this->_internalTextureFormat = internalTexFormat;
        this->_textureFormat = texFormat;
        this->_pixelValueType = pixelValType; 

        if (isMSAA_Buffer)
            this->initializeMSAABuffer();

        this->initializeStandardBuffer();
    }

    void FrameBuffer::initializeScreenQuad()
    {
        const float screenVertices[] = {
            // positions   // texCoords
            -0.3f,  1.0f,  0.0f, 1.0f,
            -0.3f,  0.7f,  0.0f, 0.0f,
             0.3f,  0.7f,  1.0f, 0.0f,
    
            -0.3f,  1.0f,  0.0f, 1.0f,
             0.3f,  0.7f,  1.0f, 0.0f,
             0.3f,  1.0f,  1.0f, 1.0f
        };

        glGenVertexArrays(1, &this->_screenQuadVAO);
        glGenBuffers(1, &this->_screenQuadVBO);
        glBindVertexArray(this->_screenQuadVAO);
        glBindBuffer(GL_ARRAY_BUFFER, this->_screenQuadVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(screenVertices), &screenVertices, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    }

    void FrameBuffer::initializeStandardBuffer()
    {
        // cudaInterop = new CUDA::Interop::CUDAInteroperationManager();

        // Generating Post Processing texture
        // ----------------------------------
        glGenTextures(1, &_postProcessedTexture);
        glBindTexture(GL_TEXTURE_2D, _postProcessedTexture);
        glTexImage2D(
                        GL_TEXTURE_2D, 
                        0, 
                        this->_internalTextureFormat, 
                        _frameBufferSize.width, 
                        _frameBufferSize.height, 
                        0, 
                        this->_textureFormat, 
                        this->_pixelValueType, 
                        NULL
                    );
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glBindTexture(GL_TEXTURE_2D, 0);
        // cudaInterop->createCUDATextureResource(cudaInterop->cudaResources[1], _postProcessedTexture, cudaGraphicsMapFlagsWriteDiscard);
        // ----------------------------------

        // framebuffer configuration
        // -------------------------
        glGenFramebuffers(1, &this->_id);
        glBindFramebuffer(GL_FRAMEBUFFER, this->_id);

        for (int i = 0; i < this->_numberOfColorAttachments; i++)
        {
            GLuint texID = 0;
            // create a color attachment texture
            glGenTextures(1, &texID);
            glBindTexture(GL_TEXTURE_2D, texID);
            glTexImage2D(
                            GL_TEXTURE_2D, 
                            0, 
                            this->_internalTextureFormat, 
                            _frameBufferSize.width, 
                            _frameBufferSize.height, 
                            0, 
                            this->_textureFormat, 
                            this->_pixelValueType, 
                            NULL
                        );
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glBindTexture(GL_TEXTURE_2D, 0);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, GL_TEXTURE_2D, texID, 0);
            _texturesID.push_back(texID);
        }

        GLenum drawBuffers[GL_MAX_COLOR_ATTACHMENTS];
        for (int i = 0; i < this->_numberOfColorAttachments; i++)
            drawBuffers[i] = GL_COLOR_ATTACHMENT0 + i;

        glDrawBuffers(this->_numberOfColorAttachments, drawBuffers);

        // TODO: Add log to console & StatusBar
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
            std::cout << "ERROR::FRAMEBUFFER:: Framebuffer is not complete!" << std::endl;
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        // Creating texture reference inside CUDA with my screen texture of COLOR_ATTACHMENT_0
        // cudaInterop->createCUDATextureResource(cudaInterop->cudaResources[0], _texturesID[0], cudaGraphicsMapFlagsReadOnly);

        _frameBufferShader->use();
        _frameBufferShader->setInt("screenTexture", 0);
    }

    void FrameBuffer::initializeMSAABuffer()
    {
        // configure MSAA framebuffer
        // --------------------------
        glGenFramebuffers(1, &this->_MSAAid);
        glBindFramebuffer(GL_FRAMEBUFFER, this->_MSAAid);

        for (int i = 0; i < this->_numberOfColorAttachments; i++)
        {
            GLuint texMultisampledID = 0;
            // create a multisampled color attachment texture
            glGenTextures(1, &texMultisampledID);
            glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, texMultisampledID);
            glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, 4, this->_internalTextureFormat, _frameBufferSize.width, _frameBufferSize.height, GL_TRUE);
            glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, 0);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, GL_TEXTURE_2D_MULTISAMPLE, texMultisampledID, 0);
            _texturesMultisampledID.push_back(texMultisampledID);
        }

        GLenum drawBuffers[GL_MAX_COLOR_ATTACHMENTS];
        for (int i = 0; i < this->_numberOfColorAttachments; i++)
            drawBuffers[i] = GL_COLOR_ATTACHMENT0 + i;
        glDrawBuffers(this->_numberOfColorAttachments, drawBuffers);

        // create a (also multisampled) renderbuffer object for depth and stencil attachments
        glGenRenderbuffers(1, &this->_renderBufferObjectID);
        glBindRenderbuffer(GL_RENDERBUFFER, this->_renderBufferObjectID);
        glRenderbufferStorageMultisample(GL_RENDERBUFFER, 4, GL_DEPTH24_STENCIL8, _frameBufferSize.width, _frameBufferSize.height);
        glBindRenderbuffer(GL_RENDERBUFFER, 0);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, this->_renderBufferObjectID);

        // TODO: Print this to console and Status Bar
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
            std::cout << "ERROR::FRAMEBUFFER:: MSAA Framebuffer is not complete! Error: " << glCheckFramebufferStatus(GL_FRAMEBUFFER) << std::endl;
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    void FrameBuffer::setNewBufferWidth(const Athena::Scalar& width)
    {
        this->_frameBufferSize.width = width;

        this->refreshFrameBufferTextureSize();
    }

    void FrameBuffer::setNewBufferHeight(const Athena::Scalar& height)
    {
        this->_frameBufferSize.height = height;

        this->refreshFrameBufferTextureSize();
    }

    void FrameBuffer::setNewBufferSize(const FrameBufferSize& newSize)
    {
        this->_frameBufferSize.width = newSize.width;
        this->_frameBufferSize.height = newSize.height;

        this->refreshFrameBufferTextureSize();
    }

    void FrameBuffer::setNewBufferSize(const Athena::Scalar& width, const Athena::Scalar& height)
    {
        this->_frameBufferSize.width = width;
        this->_frameBufferSize.height = height;

        this->refreshFrameBufferTextureSize();
    }

    void FrameBuffer::refreshFrameBufferTextureSize()
    {
        static FrameBufferSize fbSize = this->_frameBufferSize;

        // Generating Post Processing texture
        // ----------------------------------
        // cudaInterop->deleteCUDATextureResource(cudaInterop->cudaResources[1]);
        glBindTexture(GL_TEXTURE_2D, _postProcessedTexture);
        glTexImage2D(
                        GL_TEXTURE_2D, 
                        0, 
                        this->_internalTextureFormat, 
                        _frameBufferSize.width, 
                        _frameBufferSize.height, 
                        0, 
                        this->_textureFormat, 
                        this->_pixelValueType, 
                        NULL
                    );
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glBindTexture(GL_TEXTURE_2D, 0);
        // cudaInterop->createCUDATextureResource(cudaInterop->cudaResources[1], _postProcessedTexture, cudaGraphicsMapFlagsNone);
        // ----------------------------------

        //----------------------------------------------Clearing Texture Content------------------------------------------------------------------//
        // cudaInterop->deleteCUDATextureResource(cudaInterop->cudaResources[0]);
        for (int i = 0; i < this->_numberOfColorAttachments; i++)
        {
            glBindTexture(GL_TEXTURE_2D, this->_texturesID[i]);
            glTexImage2D(
                            GL_TEXTURE_2D, 
                            0, 
                            this->_internalTextureFormat, 
                            this->_frameBufferSize.width, 
                            this->_frameBufferSize.height, 
                            0, 
                            this->_textureFormat, 
                            this->_pixelValueType, 
                            NULL
                        );
            glBindTexture(GL_TEXTURE_2D, 0);
        }
        // cudaInterop->createCUDATextureResource(cudaInterop->cudaResources[0], _texturesID[0], cudaGraphicsMapFlagsReadOnly);

        for (int i = 0; i < this->_numberOfColorAttachments; i++)
        {
            glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, this->_texturesMultisampledID[i]);
            glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, 4, this->_internalTextureFormat, _frameBufferSize.width, _frameBufferSize.height, GL_TRUE);
            glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, 0);
        }

        // if (fbSize.width == this->frameBufferSize.width && fbSize.height == this->frameBufferSize.height)
            // cudaInterop->applyFilterOverTexture(_frameBufferSize.width, _frameBufferSize.height, CUDA::Interop::ScreenSpaceFilters::NONE);
        // else
            // fbSize = this->_frameBufferSize;
    }

    void FrameBuffer::bindStandardFrameBuffer()
    {
        glBindFramebuffer(GL_FRAMEBUFFER, this->_id);
        glViewport(0, 0, this->_frameBufferSize.width, this->_frameBufferSize.height);
        glEnable(GL_DEPTH_TEST); // enable depth testing (is disabled for rendering screen-space quad)

        glClearColor(refreshColor.coordinates.x, refreshColor.coordinates.y, refreshColor.coordinates.z, refreshColor.coordinates.w);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    }

    void FrameBuffer::bindMSAAFrameBuffer()
    {
        glBindTexture(GL_TEXTURE_2D, 0);
        glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, 0);
        glDisable(GL_BLEND);

        glBindFramebuffer(GL_FRAMEBUFFER, this->_MSAAid);
        glViewport(0, 0, this->_frameBufferSize.width, this->_frameBufferSize.height);
        glEnable(GL_DEPTH_TEST); // enable depth testing (is disabled for rendering screen-space quad)

        glClearColor(refreshColor.coordinates.x, refreshColor.coordinates.y, refreshColor.coordinates.z, refreshColor.coordinates.w);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
    }

    void FrameBuffer::bind()
    {
        if (this->_isMSAA_Buffer)
        {
            this->bindMSAAFrameBuffer();
        }
        else
        {
            this->blitStandardFrameBuffer();
        }
    }

    void FrameBuffer::blitStandardFrameBuffer()
    {
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glViewport(0, 0, _frameBufferSize.width, _frameBufferSize.height);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f); 
        glClear(GL_COLOR_BUFFER_BIT);
        
        glDisable(GL_DEPTH_TEST); // disable depth test so screen-space quad isn't discarded due to depth test.
    }

    void FrameBuffer::framebufferShaderCallback(const ImDrawList* dummy, const ImDrawCmd* command)
    {
        ImDrawData* draw_data = ImGui::GetDrawData();

        float L = draw_data->DisplayPos.x;
        float R = draw_data->DisplayPos.x + draw_data->DisplaySize.x;
        float T = draw_data->DisplayPos.y;
        float B = draw_data->DisplayPos.y + draw_data->DisplaySize.y;

        const float ortho_projection[4][4] =
        {
            { 2.0f/(R-L),   0.0f,         0.0f,   0.0f },
            { 0.0f,         2.0f/(T-B),   0.0f,   0.0f },
            { 0.0f,         0.0f,        -1.0f,   0.0f },
            { (R+L)/(L-R),  (T+B)/(B-T),  0.0f,   1.0f },
        };

        Athena::Matrix4 projection(ortho_projection[0][0],ortho_projection[0][1], ortho_projection[0][2], ortho_projection[0][3],
                                   ortho_projection[1][0],ortho_projection[1][1], ortho_projection[1][2], ortho_projection[1][3], 
                                   ortho_projection[2][0],ortho_projection[2][1], ortho_projection[2][2], ortho_projection[2][3],
                                   ortho_projection[3][0],ortho_projection[3][1], ortho_projection[3][2], ortho_projection[3][3]
                                );

        auto fb = (FrameBuffer*)command->UserCallbackData;
        fb->_frameBufferShader->use();
        fb->_frameBufferShader->setMat4("projection", projection);
    }

    void FrameBuffer::blitMSAAFrameBuffer()
    {
        glBindFramebuffer(GL_READ_FRAMEBUFFER, this->_MSAAid);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, this->_id);

        for (int i = 0; i < this->_numberOfColorAttachments; i++)
        {
            glReadBuffer(GL_COLOR_ATTACHMENT0 + i);
            glDrawBuffer(GL_COLOR_ATTACHMENT0 + i);
            glBlitFramebuffer(0, 0, _frameBufferSize.width, _frameBufferSize.height, 0, 0, _frameBufferSize.width, _frameBufferSize.height, GL_COLOR_BUFFER_BIT, GL_NEAREST);
        }

        glReadBuffer(GL_NONE);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        glDisable(GL_DEPTH_TEST); // disable depth test so screen-space quad isn't discarded due to depth test.

        glUseProgram(0);
        
        this->_frameBufferShader->use();
        glBindVertexArray(this->_screenQuadVAO);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, this->_texturesID[0]);

                    
        glBindTexture(GL_TEXTURE_2D, this->_postProcessedTexture);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        glUseProgram(0);
    }

    void FrameBuffer::copyAnotherFrameBuffer(const GLuint& idToCopy)
    {
        glBindFramebuffer(GL_READ_FRAMEBUFFER, idToCopy);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, this->_id);
        glBlitFramebuffer(0, 0, _frameBufferSize.width, _frameBufferSize.height, 0, 0, _frameBufferSize.width, _frameBufferSize.height, GL_COLOR_BUFFER_BIT, GL_NEAREST);

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glViewport(0, 0, _frameBufferSize.width, _frameBufferSize.height);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f); 
        glClear(GL_COLOR_BUFFER_BIT);
        
        glDisable(GL_DEPTH_TEST); // disable depth test so screen-space quad isn't discarded due to depth test.
    }

    void FrameBuffer::blit()
    {
        if (this->_isMSAA_Buffer)
        {
            this->blitMSAAFrameBuffer();
        }
        else
        {
            this->blitStandardFrameBuffer();
        }
    }

} // namespace System::Buffers
