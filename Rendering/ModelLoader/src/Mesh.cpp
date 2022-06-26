#include "../Mesh.hpp"

namespace Odysseus
{
    Mesh::Mesh(std::vector<Vertex>& vertices, std::vector<GLuint>& indices, std::vector<Texture2D>& textures) noexcept : vertices(vertices), indices(indices), textures(textures)
    {
        Mesh::setupMesh();
    }

    Mesh::Mesh(Mesh&& move) noexcept :
            vertices(std::move(move.vertices)), indices(std::move(move.indices)), textures(std::move(move.textures)),
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
            this->textures = std::move(move.textures);
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

        for(GLuint i = 0; i < textures.size(); ++i)
        {
            //activate texture
            glActiveTexture(GL_TEXTURE0 + i);

            //retreive texture infos
            std::string name;
            switch(textures[i].type)
            {
                case aiTextureType_DIFFUSE:
                    name = "diffuse" + std::to_string(diffuseIdx++);
                    break;
                case aiTextureType_SPECULAR:
                    name = "specular" + std::to_string(specularIdx++);
                    break;
                case aiTextureType_HEIGHT:
                    name = "height" + std::to_string(heightIdx++);
                    break;
                case aiTextureType_AMBIENT:
                    name = "ambient" + std::to_string(ambientIdx++);
                    break;
                default:
                    break;
            }

            //set shader uniform
            shader.setInt(name.c_str(), i);

            //bind texture
            textures[i].BindTexture();
        }
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