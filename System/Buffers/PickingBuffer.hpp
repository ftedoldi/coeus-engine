#ifndef __PICKINGBUFFER_H__
#define __PICKINGBUFFER_H__

#include <imgui.h>
#include <imgui_internal.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <IO/Input.hpp>
#include <PixelData.hpp>

#include <Scalar.hpp>
#include <Vector4.hpp>

#include <Shader.hpp>

#include <iostream>

namespace System
{
    namespace Buffers
    {
        struct PickingBufferSize {
            float width;
            float height;
        };

        class PickingBuffer
        {
            private:
                GLuint _id;
                GLuint _textureID;
                GLuint _depthTextureID;

                void init();
                
            public:
                const GLuint& id;
                const GLuint& textureID;
                const GLuint& depthTextureID;

                Odysseus::Shader* pickingShader;

                PickingBufferSize size;

                PickingBuffer();
                PickingBuffer(float width, float height);

                void enableWriting();
                void disableWriting();

                PixelData readPixel();
        };
    } // namespace Buffers
} // namespace System


#endif // __PICKINGBUFFER_H__