//#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>
#include "../Texture2D.hpp"

namespace Odysseus
{
    Texture2D::Texture2D() : width(0), height(0), internalFormat(GL_RGB), pixelDataFormat(GL_RGB), wrapS(GL_REPEAT), wrapT(GL_REPEAT), filterMin(GL_LINEAR_MIPMAP_LINEAR), filterMax(GL_LINEAR)
    {
        glGenTextures(1, &this->ID);
    }

    Texture2D::Texture2D(std::string dir, std::string path, aiTextureType type) : 
            directory(dir), path(path), type(type), width(0), height(0), 
            internalFormat(GL_RGBA), pixelDataFormat(GL_RGBA), wrapS(GL_REPEAT), 
            wrapT(GL_REPEAT), filterMin(GL_LINEAR_MIPMAP_LINEAR), filterMax(GL_LINEAR)
    {
        glGenTextures(1, &this->ID);
    }

    void Texture2D::GenerateTexture(unsigned int width, unsigned int height, unsigned char* data)
    {
        this->width = width;
        this->height = height;
        //create texture
        glBindTexture(GL_TEXTURE_2D, this->ID);
        glTexImage2D(GL_TEXTURE_2D, 0, this->internalFormat, width, height, 0, this->pixelDataFormat, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);
        //set Texture wrap and filter modes
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, this->wrapS);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, this->wrapT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, this->filterMin);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, this->filterMax);
        //unbind texture
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    void Texture2D::BindTexture() const
    {
        glBindTexture(GL_TEXTURE_2D, this->ID);
    }

    Texture2D Texture2D::loadTextureFromFile(const char* file, bool gamma)
    {
        Texture2D texture;

        int width, height, nrChannels;
        unsigned char* data = stbi_load(file, &width, &height, &nrChannels, 0);

        if(data)
        {
            if (nrChannels == 1)
            {
                texture.internalFormat = GL_RED;
                texture.pixelDataFormat = GL_RED;
            }
            else if (nrChannels == 3)
            {
                texture.internalFormat = gamma ? GL_SRGB : GL_RGB;
                texture.pixelDataFormat = GL_RGB;
            }
            else if (nrChannels == 4)
            {
                texture.internalFormat = gamma ? GL_SRGB_ALPHA : GL_RGB;
                texture.pixelDataFormat = GL_RGBA;
            }
            texture.GenerateTexture(width, height, data);
        } else
        {
            std::cout << "Image not loaded at " << texture.path << std::endl;
        }
        stbi_image_free(data);
        return texture;
    }

    //To load a file with a texture, make sure that in the same directory as the obj file
    //you have a folder called "Textures" which contains all textures used.
    void Texture2D::loadTextureFromFile(bool gamma)
    {
        int width, height, nrChannels;
        unsigned char* data = stbi_load((this->directory + "/" + "Textures" + "/" + this->path).c_str(), &width, &height, &nrChannels, 0);

        std::cout << std::endl;
        std::cout << "Directory: " << this->directory << std::endl;
        std::cout << "Path: " << this->path << std::endl;
        std::cout << "Directory + path: " << this->directory + "/" + "Textures" + "/" + this->path << std::endl;

        if(data)
        {
            if (nrChannels == 1)
            {
                this->internalFormat = GL_RED;
                this->pixelDataFormat = GL_RED;
            }
            else if (nrChannels == 3)
            {
                this->internalFormat = gamma ? GL_SRGB : GL_RGB;
                this->pixelDataFormat = GL_RGB;
            }
            else if (nrChannels == 4)
            {
                this->internalFormat = gamma ? GL_SRGB_ALPHA : GL_RGBA;
                this->pixelDataFormat = GL_RGBA;
            }
            this->GenerateTexture(width, height, data);
        } else
        {
            std::cout << "Image not loaded at " << path << std::endl;
        }
        stbi_image_free(data);
    }
}