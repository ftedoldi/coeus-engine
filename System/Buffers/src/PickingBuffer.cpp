#include "../PickingBuffer.hpp"

namespace System::Buffers
{
    
    PickingBuffer::PickingBuffer() : id(_id), textureID(_textureID), depthTextureID(_depthTextureID)
    {
        size.width = 800;
        size.height = 800;

        pickingShader = new Odysseus::Shader(".\\Shader\\picking.vert", ".\\Shader\\picking.frag");

        init();
    }

    PickingBuffer::PickingBuffer(float width, float height) : id(_id), textureID(_textureID), depthTextureID(_depthTextureID)
    {
        size.width = width;
        size.height = height;

        pickingShader = new Odysseus::Shader(".\\Shader\\picking.vert", ".\\Shader\\picking.frag");

        init();
    }

    void PickingBuffer::init() 
    {
        // Create the FBO
        glGenFramebuffers(1, &this->_id);
        glBindFramebuffer(GL_FRAMEBUFFER, _id);

        // Create the texture object for the primitive information buffer
        glGenTextures(1, &_textureID);
        glBindTexture(GL_TEXTURE_2D, _textureID);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, size.width, size.height,
                    0, GL_RGB, GL_FLOAT, NULL);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                    _textureID, 0);

        // Create the texture object for the depth buffer
        glGenTextures(1, &_depthTextureID);
        glBindTexture(GL_TEXTURE_2D, _depthTextureID);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, size.width, size.height,
                    0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D,
                    _depthTextureID, 0);

        // Disable reading to avoid problems with older GPUs
        glReadBuffer(GL_NONE);

        glDrawBuffer(GL_COLOR_ATTACHMENT0);

        // Verify that the FBO is correct
        GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);

        if (status != GL_FRAMEBUFFER_COMPLETE) {
            std::cout << "Picking FrameBuffer error, status: 0x" << status << std::endl;
        }

        // Restore the default framebuffer
        glBindTexture(GL_TEXTURE_2D, 0);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    void PickingBuffer::enableWriting()
    {
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, this->_id);
    }

    void PickingBuffer::disableWriting()
    {
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
    }

    PixelData PickingBuffer::readPixel()
    {
        glBindFramebuffer(GL_READ_FRAMEBUFFER, this->_id);
        glReadBuffer(GL_COLOR_ATTACHMENT0);

        PixelData pixel;
        glReadPixels(Input::mouse.xPosition, Input::mouse.yPosition, 1, 1, GL_RGB, GL_FLOAT, &pixel);

        glReadBuffer(GL_NONE);
        glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);

        return pixel;
    }

}