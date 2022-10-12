#include "../Mesh.hpp"

#include <Window.hpp>

#include <Component.hpp>
#include <EditorCamera.hpp>

#include <Texture2D.hpp>
#include <Folder.hpp>

#include <LightInfo.hpp>

// TODO: Test if everythin works with PBR Materials
namespace Odysseus
{
    Mesh::Mesh()
    {
        std::cout << "Mesh created" << std::endl;

        // TODO: Generate random int
        std::random_device rd;
        std::mt19937 rng(rd());
        std::uniform_int_distribution<int> uni;

        this->_uniqueID = uni(rng);
        this->_uniqueFloatID = static_cast<float>(this->_uniqueID);

        this->_editorTextureID = Odysseus::Texture2D::loadTextureFromFile(
                                                                            (System::Folder::getFolderPath("Icons").string() + "/mesh.png").c_str(), 
                                                                            true
                                                                        ).ID;;
        this->_hasEditorTexture = true;
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
        }
        else
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
        
        LightInfo::computeLighting(this->shader);

        System::Picking::PickableObject::insertPickableObject(this->_uniqueFloatID, this->sceneObject);

        auto worldPosition = Transform::GetWorldTransform(this->transform, this->transform);
        auto tmp = Odysseus::SceneManager::activeScene->sceneEditor->editorCamera->getViewTransform(worldPosition);

        this->shader->setVec3("viewPos", Odysseus::SceneManager::activeScene->sceneEditor->editorCamera->transform->position);
        this->shader->setVec3("WorldPosition", worldPosition->position);
        this->shader->setVec4("WorldRotation", worldPosition->rotation.asVector4());
        this->shader->setVec3("WorldScale", worldPosition->localScale);

        this->shader->setVec3("position", tmp->position);
        this->shader->setVec4("rotation", tmp->rotation.asVector4());
        this->shader->setVec3("scale", tmp->localScale);

        this->shader->setFloat("ID", this->_uniqueFloatID);

        Athena::Matrix4 projection = Odysseus::EditorCamera::perspective(
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

        LightInfo::computeLighting(this->shader);

        auto worldPosition = Transform::GetWorldTransform(this->transform, this->transform);
        auto tmp = Odysseus::SceneManager::activeScene->sceneEditor->editorCamera->getViewTransform(worldPosition);

        this->shader->setVec3("viewPos", Odysseus::SceneManager::activeScene->sceneEditor->editorCamera->transform->position);
        this->shader->setVec3("WorldPosition", worldPosition->position);
        this->shader->setVec4("WorldRotation", worldPosition->rotation.asVector4());
        this->shader->setVec3("WorldScale", worldPosition->localScale);

        this->shader->setVec3("position", tmp->position);
        this->shader->setVec4("rotation", tmp->rotation.asVector4());
        this->shader->setVec3("scale", tmp->localScale);

        //TODO: Swap uniform set with attribute set -> ID is more an attribute than a Uniform variable
        //TODO: Find a way to pass the UUID instead of this useless value
        this->shader->setFloat("ID", this->_uniqueFloatID);

        //TODO: call this inside framebuffer callback to avoid creating a perspective even if not needed
        Athena::Matrix4 projection = Odysseus::EditorCamera::perspective(
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

    int Mesh::getUniqueID()
    {
        return this->_uniqueID;
    }

    std::string Mesh::toString()
    {
        return "Mesh";
    }

    void Mesh::showComponentFieldsInEditor()
    {
        std::string numOfVertices("Number of Vertices: " + std::to_string(this->vertices.size()));
        ImGui::Text(numOfVertices.c_str());
        if (this->_isPBR)
        {
            ImGui::Text("Physics Material Properties:");
            float albedo[] = { 
                                physicsMaterial.albedo.coordinates.x, 
                                physicsMaterial.albedo.coordinates.y, 
                                physicsMaterial.albedo.coordinates.z 
                            };
            ImGui::InputFloat3("Albedo", albedo);
            physicsMaterial.albedo = Athena::Vector3(albedo[0], albedo[1], albedo[2]);
            ImGui::InputFloat("Metallic", &physicsMaterial.metallic);
            ImGui::InputFloat("Roughness", &physicsMaterial.roughness);
            ImGui::InputFloat("Ambient Occlusion Factor", &physicsMaterial.AO);
            // TODO: Add a way to add PBR textures to the textures array
        }
        else
        {
            ImGui::Text("Phong Material Properties:");
            float diffuse[] = { 
                                phongMaterial.Diffuse.coordinates.x, 
                                phongMaterial.Diffuse.coordinates.y, 
                                phongMaterial.Diffuse.coordinates.z 
                            };
            ImGui::InputFloat3("Diffuse Color", diffuse);
            phongMaterial.Diffuse = Athena::Vector3(diffuse[0], diffuse[1], diffuse[2]);
            float specular[] = { 
                                phongMaterial.Specular.coordinates.x, 
                                phongMaterial.Specular.coordinates.y, 
                                phongMaterial.Specular.coordinates.z 
                            };
            ImGui::InputFloat3("Specular Color", specular);
            phongMaterial.Specular = Athena::Vector3(specular[0], specular[1], specular[2]);
            float ambient[] = { 
                                phongMaterial.Ambient.coordinates.x, 
                                phongMaterial.Ambient.coordinates.y, 
                                phongMaterial.Ambient.coordinates.z 
                            };
            ImGui::InputFloat3("Ambient Color", ambient);
            phongMaterial.Ambient = Athena::Vector3(ambient[0], ambient[1], ambient[2]);
            ImGui::InputFloat("Shininess", &phongMaterial.Shininess);
            // TODO: Add a way to add PBR textures to the textures array
        }
    }

    void Mesh::serialize(YAML::Emitter& out)
    {
        YAML::Emitter vert;

        vert << YAML::BeginMap;
            vert << YAML::Key << "Scene Object" << YAML::Value << this->sceneObject->ID;
            vert << YAML::Key << "Vertices";
            vert << YAML::BeginSeq;
                for (auto vertex : this->vertices)
                {
                    vert << YAML::BeginMap;
                        vert << YAML::Key << "Position";
                        vert << YAML::BeginMap;
                            vert << YAML::Key << "X" << YAML::Value << vertex.Position.coordinates.x;
                            vert << YAML::Key << "Y" << YAML::Value << vertex.Position.coordinates.y;
                            vert << YAML::Key << "Z" << YAML::Value << vertex.Position.coordinates.z;
                        vert << YAML::EndMap;
                        vert << YAML::Key << "Normal";
                        vert << YAML::BeginMap;
                            vert << YAML::Key << "X" << YAML::Value << vertex.Normal.coordinates.x;
                            vert << YAML::Key << "Y" << YAML::Value << vertex.Normal.coordinates.y;
                            vert << YAML::Key << "Z" << YAML::Value << vertex.Normal.coordinates.z;
                        vert << YAML::EndMap;
                        vert << YAML::Key << "Tangent";
                        vert << YAML::BeginMap;
                            vert << YAML::Key << "X" << YAML::Value << vertex.Tangent.coordinates.x;
                            vert << YAML::Key << "Y" << YAML::Value << vertex.Tangent.coordinates.y;
                            vert << YAML::Key << "Z" << YAML::Value << vertex.Tangent.coordinates.z;
                        vert << YAML::EndMap;
                        vert << YAML::Key << "Texture Coordinates";
                        vert << YAML::BeginMap;
                            vert << YAML::Key << "X" << YAML::Value << vertex.TexCoords.coordinates.x;
                            vert << YAML::Key << "Y" << YAML::Value << vertex.TexCoords.coordinates.y;
                        vert << YAML::EndMap;
                    vert << YAML::EndMap;
                }
            vert << YAML::EndSeq;
            vert << YAML::Key << "Indices";
            vert << YAML::BeginSeq;
                for (auto idx : this->indices)
                {
                    vert << YAML::BeginMap;
                        vert << YAML::Key << "Index" << YAML::Value << idx;
                    vert << YAML::EndMap;
                }
            vert << YAML::EndSeq;
        vert << YAML::EndMap;

        std::string filePath("./Assets/Scenes/Meshes/" + std::to_string(this->sceneObject->ID) + ".meta");
        std::ofstream fout(filePath);
        fout << vert.c_str();
        fout.close();

        out << YAML::Key << this->toString();
        out << YAML::BeginMap;
            out << YAML::Key << "Vertices Path" << YAML::Value << filePath;
            out << YAML::Key << "Is PBR" << YAML::Value << this->_isPBR;
            if (this->_isPBR)
            {
                out << YAML::Key << "Albedo";
                out << YAML::BeginMap;
                    out << YAML::Key << "Red" << YAML::Value << physicsMaterial.albedo.coordinates.x;
                    out << YAML::Key << "Green" << YAML::Value << physicsMaterial.albedo.coordinates.y;
                    out << YAML::Key << "Blue" << YAML::Value << physicsMaterial.albedo.coordinates.z;
                out << YAML::EndMap;
                out << YAML::Key << "Metallic" << YAML::Value << physicsMaterial.metallic;
                out << YAML::Key << "Roughness" << YAML::Value << physicsMaterial.roughness;
                out << YAML::Key << "Ambient Occlusion" << YAML::Value << physicsMaterial.AO;
                out << YAML::Key << "PBR Textures";
                out << YAML::BeginSeq;
                    for (auto tex : physicsMaterial.PBR_textures)
                    {
                        out << YAML::BeginMap;
                            out << YAML::Key << "Directory" << YAML::Value << tex.directory;
                            out << YAML::Key << "Path" << YAML::Value << tex.path;
                            out << YAML::Key << "Type" << YAML::Value << tex.type;
                        out << YAML::EndMap;
                    }
                out << YAML::EndSeq;
            }
            else
            {
                out << YAML::Key << "Diffuse";
                out << YAML::BeginMap;
                    out << YAML::Key << "Red" << YAML::Value << phongMaterial.Diffuse.coordinates.x;
                    out << YAML::Key << "Green" << YAML::Value << phongMaterial.Diffuse.coordinates.y;
                    out << YAML::Key << "Blue" << YAML::Value << phongMaterial.Diffuse.coordinates.z;
                out << YAML::EndMap;
                out << YAML::Key << "Specular";
                out << YAML::BeginMap;
                    out << YAML::Key << "Red" << YAML::Value << phongMaterial.Specular.coordinates.x;
                    out << YAML::Key << "Green" << YAML::Value << phongMaterial.Specular.coordinates.y;
                    out << YAML::Key << "Blue" << YAML::Value << phongMaterial.Specular.coordinates.z;
                out << YAML::EndMap;
                out << YAML::Key << "Ambient";
                out << YAML::BeginMap;
                    out << YAML::Key << "Red" << YAML::Value << phongMaterial.Ambient.coordinates.x;
                    out << YAML::Key << "Green" << YAML::Value << phongMaterial.Ambient.coordinates.y;
                    out << YAML::Key << "Blue" << YAML::Value << phongMaterial.Ambient.coordinates.z;
                out << YAML::EndMap;
                out << YAML::Key << "Shininess" << YAML::Value << phongMaterial.Shininess;
                out << YAML::Key << "Textures";
                out << YAML::BeginSeq;
                    for (auto tex : phongMaterial.Textures)
                    {
                        out << YAML::BeginMap;
                            out << YAML::Key << "Directory" << YAML::Value << tex.directory;
                            out << YAML::Key << "Path" << YAML::Value << tex.path;
                            out << YAML::Key << "Type" << YAML::Value << tex.type;
                        out << YAML::EndMap;
                    }
                out << YAML::EndSeq;
            }
            out << YAML::Key << "Vertex Shader Path" << YAML::Value << this->shader->vertexShaderPath;
            out << YAML::Key << "Fragment Shader Path" << YAML::Value << this->shader->fragmentShaderPath;
        out << YAML::EndMap;
    }

    System::Component* Mesh::deserialize(YAML::Node& node)
    {
        auto component = node[this->toString()];

        auto verticesPath = component["Vertices Path"].as<std::string>();
        
        std::ifstream stream(verticesPath);
        std::stringstream strStream;
        strStream << stream.rdbuf();

        YAML::Node data = YAML::Load(strStream.str());

        if (!data["Scene Object"])
        {
            std::cerr << "Could not load the Vertices File at path: " << verticesPath << std::endl;
            return nullptr;
        }

        // this->sceneObject->ID = data["Scene Object"].as<uint64_t>();

        auto verticesData = data["Vertices"];
        for (auto v : verticesData)
        {
            Vertex deserializedVertex;
            deserializedVertex.Position = Athena::Vector3(v["Position"]["X"].as<float>(), v["Position"]["Y"].as<float>(), v["Position"]["Z"].as<float>());
            deserializedVertex.Normal = Athena::Vector3(v["Normal"]["X"].as<float>(), v["Normal"]["Y"].as<float>(), v["Normal"]["Z"].as<float>());
            deserializedVertex.Tangent = Athena::Vector3(v["Tangent"]["X"].as<float>(), v["Tangent"]["Y"].as<float>(), v["Tangent"]["Z"].as<float>());
            deserializedVertex.TexCoords = Athena::Vector2(v["Texture Coordinates"]["X"].as<float>(), v["Texture Coordinates"]["Y"].as<float>());

            this->vertices.push_back(deserializedVertex);
        }

        auto indicesData = data["Indices"];
        for (auto i : indicesData)
        {
            GLuint deserializedIndex = i["Index"].as<GLuint>();

            this->indices.push_back(deserializedIndex);
        }

        auto vShaderPath = component["Vertex Shader Path"].as<std::string>();
        auto fShaderPath = component["Fragment Shader Path"].as<std::string>();

        this->shader = new Odysseus::Shader(vShaderPath.c_str(), fShaderPath.c_str());

        this->_isPBR = component["Is PBR"].as<bool>();

        // TODO: Deserializa materials
        if (this->_isPBR)
        {
            this->physicsMaterial = PhysicsMaterial();

            this->physicsMaterial.albedo = Athena::Vector3(
                                                            component["Albedo"]["Red"].as<float>(),
                                                            component["Albedo"]["Green"].as<float>(),
                                                            component["Albedo"]["Blue"].as<float>()
                                                        );
            this->physicsMaterial.metallic = component["Metallic"].as<float>();
            this->physicsMaterial.roughness = component["Roughness"].as<float>();
            this->physicsMaterial.AO = component["Ambient Occlusion"].as<float>();

            auto pbrMatTextures = component["PBR Textures"];
            for (auto t : pbrMatTextures)
            {
                std::string textureDirectory = t["Directory"].as<std::string>();
                std::string texturePath = t["Path"].as<std::string>();
                aiTextureType textureType = static_cast<aiTextureType>(t["Type"].as<int>());

                Texture2D physicsMaterialTexture = Texture2D(textureDirectory, texturePath, textureType);
                physicsMaterialTexture.loadTextureFromFile(true);
                this->physicsMaterial.PBR_textures.push_back(physicsMaterialTexture);
            }
        }
        else
        {
            this->phongMaterial = PhongMaterial();

            this->phongMaterial.Diffuse = Athena::Vector3(
                                                            component["Diffuse"]["Red"].as<float>(),
                                                            component["Diffuse"]["Green"].as<float>(),
                                                            component["Diffuse"]["Blue"].as<float>()
                                                        );
            this->phongMaterial.Specular = Athena::Vector3(
                                                            component["Specular"]["Red"].as<float>(),
                                                            component["Specular"]["Green"].as<float>(),
                                                            component["Specular"]["Blue"].as<float>()
                                                        );
            this->phongMaterial.Ambient = Athena::Vector3(
                                                            component["Ambient"]["Red"].as<float>(),
                                                            component["Ambient"]["Green"].as<float>(),
                                                            component["Ambient"]["Blue"].as<float>()
                                                        );
            this->phongMaterial.Shininess = component["Shininess"].as<float>();

            auto phongMatTextures = component["Textures"];

            for (auto t : phongMatTextures)
            {
                std::string textureDirectory = t["Directory"].as<std::string>();
                std::string texturePath = t["Path"].as<std::string>();
                aiTextureType textureType = static_cast<aiTextureType>(t["Type"].as<int>());
                std::cout << "TEXTURE TYPE: " << textureType << std::endl;

                Texture2D phongMaterialTexture = Texture2D(textureDirectory, texturePath, textureType);
                phongMaterialTexture.loadTextureFromFile(true);
                this->phongMaterial.Textures.push_back(phongMaterialTexture);
            }
        }

        return this;
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

    SERIALIZE_CLASS
    {
        System::SerializableClass::registerClass<Mesh>("Mesh");
    }
}