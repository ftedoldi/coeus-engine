#ifndef Texture2D_HPP
#define Texture2D_HPP

#include <glad/glad.h>
#include <assimp/scene.h>

#include <iostream>

namespace Odysseus
{
    class Texture2D
    {
    public:
        //Holds the ID of the texture 
        unsigned int ID;
        std::string directory;
        std::string path;
        aiTextureType type;
        //Holds the dimension of the texture
        unsigned int width, height;
        //Specifies the number of color components in a texture
        unsigned int internalFormat;
        //Specifies the format of the pixel data
        unsigned int pixelDataFormat;
        //Texture configuration
        unsigned int wrapS; //wrapping on S axis
        unsigned int wrapT; //wrapping on T axis
        unsigned int filterMin; //filtering mode if texture pixels < screen pixels
        unsigned int filterMax; //filtering mode if texture pixels > screen pixels

        //default constructor
        Texture2D();

        Texture2D(std::string dir, std::string path, aiTextureType type);
        //generate a texture from the image passed as parameter
        void GenerateTexture(unsigned int width, unsigned int height, unsigned char* data);
        //Binds the texture as current active texture
        void BindTexture() const;

        static Texture2D loadTextureFromFile(const char *file);
        void loadTextureFromFile();

    };
}
#endif