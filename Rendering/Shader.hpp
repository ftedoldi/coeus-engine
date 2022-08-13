#ifndef SHADER_HPP
#define SHADER_HPP

#include <glad/glad.h>
#include <coeus.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

namespace Odysseus
{
    class Shader
    {
    public:
        std::string vertexShaderPath, fragmentShaderPath;

        unsigned int ID;
        
        Shader();
        // constructor generates the shader on the fly
        Shader(const char* vertexPath, const char* fragmentPath);

        void assignShadersPath(const char* vertexPath, const char* fragmentPath);
        Shader* newShaderAtPath(const char* vertexPath, const char* fragmentPath);
        // activate the shader
        // ------------------------------------------------------------------------
        void use() const;
        // utility uniform functions
        // ------------------------------------------------------------------------
        void setBool(const std::string &name, bool value) const;
        // ------------------------------------------------------------------------
        void setInt(const std::string &name, int value) const;
        // ------------------------------------------------------------------------
        void setFloat(const std::string &name, float value) const;
        // ------------------------------------------------------------------------
        void setVec2(const std::string &name, const Athena::Vector2& value) const;
        void setVec2(const std::string &name, float x, float y) const;
        // ------------------------------------------------------------------------
        void setVec3(const std::string &name, const Athena::Vector3& value) const;
        void setVec3(const std::string &name, float x, float y, float z) const;
        // ------------------------------------------------------------------------
        void setVec4(const std::string &name, const Athena::Vector4& value) const;
        void setVec4(const std::string &name, float x, float y, float z, float w) const;
        // ------------------------------------------------------------------------
        void setMat2(const std::string &name, const Athena::Matrix2& mat) const;
        // ------------------------------------------------------------------------
        void setMat3(const std::string &name, const Athena::Matrix3& mat) const;
        // ------------------------------------------------------------------------
        void setMat4(const std::string &name, const Athena::Matrix4& mat) const;

    private:
        // utility function for checking shader compilation/linking errors.
        // ------------------------------------------------------------------------
        void checkCompileErrors(GLuint shader, std::string type);
    };
}
#endif