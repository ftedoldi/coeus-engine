#ifndef __FRAMEBUFFER_H__
#define __FRAMEBUFFER_H__

#include <imgui.h>
#include <imgui_internal.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <Scalar.hpp>
#include <Vector4.hpp>

#include <Shader.hpp>

namespace System {
    namespace Buffers {
        struct FrameBufferSize {
            int width;
            int height;
        };

        class FrameBuffer {
            private:
                GLuint _id;
                GLuint _MSAAid;
                GLuint _textureID;
                GLuint _textureMultisampleID;
                GLuint _renderBufferObjectID;

                bool _isMSAA_Buffer;

                FrameBufferSize _frameBufferSize;

                GLuint _screenQuadVAO;
                GLuint _screenQuadVBO;

                Odysseus::Shader* _frameBufferShader;

                void initializeScreenQuad();

                void initializeStandardBuffer();
                void initializeMSAABuffer();

                void bindStandardFrameBuffer();
                void bindMSAAFrameBuffer();

                void blitStandardFrameBuffer();
                void blitMSAAFrameBuffer();

            public:
                const GLuint& ID;
                const GLuint& MSAA_ID;
                const GLuint& textureID;
                const GLuint& textureMultisampleID;
                const GLuint& renderBufferObjectID;

                const FrameBufferSize& frameBufferSize;

                const Odysseus::Shader* frameBufferShader;

                Athena::Vector4 refreshColor;

                FrameBuffer();
                FrameBuffer(
                                const Athena::Scalar& width, 
                                const Athena::Scalar& height, 
                                bool isMSAA_Buffer = false, 
                                Athena::Vector4 refreshColor = Athena::Vector4(0.4f, 0.2f, 0.6f, 0.5f)
                            );

                void setNewBufferWidth(const Athena::Scalar& width);
                void setNewBufferHeight(const Athena::Scalar& height);

                void setNewBufferSize(const FrameBufferSize& newSize);
                void setNewBufferSize(const Athena::Scalar& width, const Athena::Scalar& height);

                void refreshFrameBufferTextureSize();

                void copyAnotherFrameBuffer(const GLuint& idToCopy);

                static void framebufferShaderCallback(const ImDrawList*, const ImDrawCmd* command);

                void bind();
                void blit();
        };
    }
}

#endif // __FRAMEBUFFER_H__