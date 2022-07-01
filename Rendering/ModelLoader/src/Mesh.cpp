#include "../Mesh.hpp"

namespace Odysseus
{

    Mesh::Mesh(std::vector<Vertex>& vertices, std::vector<GLuint>& indices, Material& mat) noexcept 
    : vertices(vertices), indices(indices), material(mat)
    {
        Mesh::setupMesh();
    }

    Mesh::Mesh(Mesh&& move) noexcept :
            vertices(std::move(move.vertices)), indices(std::move(move.indices)), material(std::move(move.material)),
            VAO(move.VAO), VBO(move.VBO), EBO(move.EBO)
    {
        move.VAO = 0;
        move.VBO = 0;
        move.EBO = 0;
    }

    Mesh& Mesh::operator=(Mesh&& move) noexcept
    {
        if(&move == this)
            return *this;
        
        //firstly i delete all the resources that this mesh is pointing to
        Mesh::freeGPUresources();

        //if move has resources
        if(move.VAO)
        {
            this->vertices = std::move(move.vertices);
            this->indices = std::move(move.indices);
            this->material = std::move(move.material);
            this->VAO = move.VAO;
            this->VBO = move.VBO;
            this->EBO = move.EBO;

            move.VAO = 0;
            move.VBO = 0;
            move.EBO = 0;
        }
        else //move doesnt have resources
        {
            this->VAO = 0;
            this->VBO = 0;
            this->EBO = 0;
        }

        return *this;
    }

    Mesh::~Mesh() noexcept
    {
        Mesh::freeGPUresources();
    }

    void Mesh::Draw(Shader &shader)
    {
        GLuint diffuseIdx = 0;
        GLuint specularIdx = 0;
        GLuint heightIdx = 0;
        GLuint ambientIdx = 0;

        //call to material

        if(this->material.Textures.size() > 0)
            material.loadShaderTexture(shader);
        else
            material.loadShaderMaterial(shader);
        
        // draw mesh
        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, static_cast<GLuint>(indices.size()), GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);

        // set everything back to default
        glActiveTexture(GL_TEXTURE0);
    }

    void Mesh::setupMesh()
    {
        //creating buffers
        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);
        glGenBuffers(1, &EBO);
    
        //bind the active VAO and VBO buffer
        glBindVertexArray(VAO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);

        //allocates memory inside the currently active buffer object (VBO)
        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), &vertices[0], GL_STATIC_DRAW);  
        
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), 
                    &indices[0], GL_STATIC_DRAW);

        // vertex positions
        glEnableVertexAttribArray(0);	
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
        // vertex normals
        glEnableVertexAttribArray(1);	
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, Normal));
        // vertex texture coords
        glEnableVertexAttribArray(2);	
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, TexCoords));
        // vertex tangent
        glEnableVertexAttribArray(3);
        glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, Tangent));
        // vertex bitangent
        glEnableVertexAttribArray(4);
        glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, Bitangent));

        glBindVertexArray(0);
    }

    void Mesh::freeGPUresources()
    {
        if(this->VAO)
        {
            glDeleteVertexArrays(1, &this->VAO);
            glDeleteBuffers(1, &this->VBO);
            glDeleteBuffers(1, &this->EBO);
        }
    }
}