#include "../Mesh.hpp"

namespace Odysseus
{
    Mesh::Mesh()
    {
        std::cout << "Mesh created" << std::endl;
    }

    void Mesh::start()
    {
        this->setupMesh();

        this->shader->use();

        if(this->_isPBR)
        {
            if(this->physicsMaterial.PBR_textures.size() > 0)
            {
                physicsMaterial.loadShaderTexture(this->shader);
            }
            else
            {
                physicsMaterial.loadShaderMaterial(this->shader);
            }
        }else
        {
            if(this->phongMaterial.Textures.size() > 0)
            {
                phongMaterial.loadShaderTexture(this->shader);
            }
            else
            {
                phongMaterial.loadShaderMaterial(this->shader);
            }
        }

        auto tmp = Odysseus::Camera::main->getViewTransform(Transform::GetWorldTransform(this->transform, this->transform));

        this->shader->setVec3("viewPos", Odysseus::Camera::main->transform->position);
        this->shader->setVec3("WorldPosition", this->transform->position);
        this->shader->setVec4("WorldRotation", this->transform->rotation.asVector4());
        this->shader->setVec3("WorldScale", this->transform->localScale);

        this->shader->setVec3("position", tmp->position);
        this->shader->setVec4("rotation", tmp->rotation.asVector4());
        this->shader->setVec3("scale", tmp->localScale);

        Athena::Matrix4 projection = Odysseus::Camera::perspective(
                                                                    45.0f, 
                                                                    System::Window::sceneFrameBuffer->frameBufferSize.width / System::Window::sceneFrameBuffer->frameBufferSize.height, 
                                                                    0.1f, 
                                                                    100.0f
                                                                );
        projection.data[0] = projection.data[0] / (System::Window::sceneFrameBuffer->frameBufferSize.width / (float)System::Window::sceneFrameBuffer->frameBufferSize.height);
        projection.data[5] = projection.data[0];

        this->shader->setMat4("projection", projection);
    }

    void Mesh::update()
    {
        this->shader->use();

        if(this->_isPBR)
        {
            if(this->physicsMaterial.PBR_textures.size() > 0)
            {
                physicsMaterial.loadShaderTexture(this->shader);
            }
            else
            {
                physicsMaterial.loadShaderMaterial(this->shader);
            }
        }else
        {
            if(this->phongMaterial.Textures.size() > 0)
            {
                phongMaterial.loadShaderTexture(this->shader);
            }
            else
            {
                phongMaterial.loadShaderMaterial(this->shader);
            }

        }

        auto worldPosition = Transform::GetWorldTransform(this->transform, this->transform);
        auto tmp = Odysseus::Camera::main->getViewTransform(worldPosition);

        this->shader->setVec3("viewPos", Odysseus::Camera::main->transform->position);
        this->shader->setVec3("WorldPosition", worldPosition->position);
        this->shader->setVec4("WorldRotation", worldPosition->rotation.asVector4());
        this->shader->setVec3("WorldScale", worldPosition->localScale);

        this->shader->setVec3("position", tmp->position);
        this->shader->setVec4("rotation", tmp->rotation.asVector4());
        this->shader->setVec3("scale", tmp->localScale);

        //TODO: call this inside framebuffer callback to avoid creating a perspective even if not needed
        Athena::Matrix4 projection = Odysseus::Camera::perspective(
                                                                    45.0f, 
                                                                    System::Window::sceneFrameBuffer->frameBufferSize.width / System::Window::sceneFrameBuffer->frameBufferSize.height, 
                                                                    0.1f, 
                                                                    100.0f
                                                                );
        projection.data[0] = projection.data[0] / (System::Window::sceneFrameBuffer->frameBufferSize.width / (float)System::Window::sceneFrameBuffer->frameBufferSize.height);
        projection.data[5] = projection.data[0];

        this->shader->setMat4("projection", projection);
        
        // draw mesh
        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, static_cast<GLuint>(this->indices.size()), GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);

        // set everything back to default
        glActiveTexture(GL_TEXTURE0);
    }

    void Mesh::setOrderOfExecution(const short& newOrderOfExecution)
    {

    }

    short Mesh::getUniqueID()
    {
        return this->_uniqueID;
    }

    std::string Mesh::toString()
    {
        return "Mesh";
    }

    void Mesh::setVertices(std::vector<Vertex>& vertices)
    {
        this->vertices = vertices;
    }
    void Mesh::setIndices(std::vector<GLuint>& indices)
    {
        this->indices = indices;
    }

    void Mesh::setPhongMaterial(PhongMaterial& mat)
    {
        this->phongMaterial = mat;
    }

    void Mesh::setPhysicsMaterial(PhysicsMaterial& mat)
    {
        this->physicsMaterial = mat;
    }

    void Mesh::setShader(Shader* shader)
    {
        this->shader = shader;
    }

    Mesh::~Mesh() noexcept
    {
        Mesh::freeGPUresources();
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

        glBindVertexArray(0);
    }

    void Mesh::setIfPBR(bool isPBR)
    {
        this->_isPBR = isPBR;
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