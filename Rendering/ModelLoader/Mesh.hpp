#ifndef MESH_HPP
#define MESH_HPP

#include <coeus.hpp>
#include <shader.hpp>
#include <Component.hpp> 
#include <Camera.hpp> 
#include <vector>
#include "../Texture/Texture2D.hpp"
#include <PhongMaterial.hpp>
#include <PhysicsMaterial.hpp>
#include <Time.hpp>

namespace Odysseus
{
    struct Vertex
    {
        Athena::Vector3 Position;
        Athena::Vector3 Normal;
        Athena::Vector2 TexCoords;
        Athena::Vector3 Tangent;
    };
    
    class Mesh : public System::Component
    {
    public:
        std::vector<Vertex> vertices;
        std::vector<GLuint> indices;
        PhongMaterial phongMaterial;
        PhysicsMaterial physicsMaterial;

        Shader* shader;
        GLuint VAO;

        Mesh();
        ~Mesh() noexcept;

        void setVertices(std::vector<Vertex>& vertices);
        void setIndices(std::vector<GLuint>& indices);
        void setPhongMaterial(PhongMaterial& mat);
        void setPhysicsMaterial(PhysicsMaterial& mat);
        void setShader(Shader* shader);

        virtual void start();
        virtual void update();

        virtual void setOrderOfExecution(const short& newOrderOfExecution);

        virtual short getUniqueID();

        virtual std::string toString();

    private:

        GLuint VBO, EBO;

        bool hasTexture;

        //Inizialize VAO, VBO, EBO
        void setupMesh();

        void freeGPUresources();

    };
}
#endif