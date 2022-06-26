#ifndef MESH_HPP
#define MESH_HPP

#include <coeus.hpp>
#include <shader.hpp>
#include <vector>
#include "../Texture/Texture2D.hpp"

namespace Odysseus
{
    struct Vertex
    {
        Athena::Vector3 Position;
        Athena::Vector3 Normal;
        Athena::Vector2 TexCoords;
        Athena::Vector3 Tangent;
        Athena::Vector3 Bitangent;
    };

    /*struct Material
    {
        Athena::Vector3 Diffuse;
        Athena::Vector3 Specular;
        Athena::Vector3 Ambient;
        float Shininess;
    };*/

    class Mesh
    {
    public:

        std::vector<Vertex> vertices;
        std::vector<GLuint> indices;
        std::vector<Texture2D> textures;
        GLuint VAO;

        //Using move semantics, we delete the copy constructor and copy operator=
        Mesh(const Mesh& mesh) = delete;
        Mesh& operator=(const Mesh& mesh) = delete;

        Mesh(std::vector<Vertex>& vertices, std::vector<GLuint>& indices, std::vector<Texture2D>& textures) noexcept;

        //Creating the move constructor and move assignment
        Mesh(Mesh&& move) noexcept;
        Mesh& operator=(Mesh&& move) noexcept;

        //Destructor to delete dynamically allocated resources
        ~Mesh() noexcept;

        //Render of the mesh
        void Draw(Shader &shader);

    private:

        GLuint VBO, EBO;

        //Inizialize VAO, VBO, EBO
        void setupMesh();

        void freeGPUresources();

    };
}
#endif