#include "../Model.hpp"

#include <ModelBase.hpp>

namespace Odysseus
{

    Model::Model(const std::string& path, Shader* textureShader, Shader* materialShader, bool isPBR, const std::string& objectType) :
            textureShader(textureShader), materialShader(materialShader), objectType(objectType)
    {
        this->_isPBR = isPBR;
        loadModel(path);
    }

    // TODO: Set the position of a model in the world correctly
    void Model::loadModel(const std::string& path)
    {
        Assimp::Importer importer;
        const aiScene* scene = importer.ReadFile(path, aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_JoinIdenticalVertices | aiProcess_CalcTangentSpace /*| aiProcess_FlipUVs*/);
        //checking for errors in the scene creation
        if(!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
        {
            std::cout << "ASSIMP SCENE ERROR: " << importer.GetErrorString() << std::endl;
            return;
        }

        this->directory = path.substr(0, path.find_last_of('\\'));

        auto name = path.substr(path.find_last_of('\\') + 1);
        name = name.substr(0, name.find_last_of('.'));

        processNode(scene->mRootNode, scene, name);
    }

    void Model::processMeshNode(aiNode* node, const aiScene* scene, Transform* parent)
    {
        if(node->mNumMeshes != 0)
        {
            Odysseus::SceneObject* obj = new SceneObject();
            objectsCreated.push_back(obj);

            for(GLuint i = 0; i < node->mNumMeshes; ++i)
            {
                obj->transform->name = "Mesh";
                Athena::Vector3 scale;
                Athena::Vector3 position;
                Athena::Quaternion rotation;
                if(node->mParent != nullptr)
                {
                    //local transformation matrix of the node
                    aiMatrix4x4 transform =  node->mTransformation;
                    
                    Athena::Matrix4 AthenaTransform((Athena::Scalar)transform.a1, (Athena::Scalar)transform.a2, (Athena::Scalar)transform.a3, (Athena::Scalar)transform.d1,
                                                    (Athena::Scalar)transform.b1, (Athena::Scalar)transform.b2, (Athena::Scalar)transform.b3, (Athena::Scalar)transform.d2,
                                                    (Athena::Scalar)transform.c1, (Athena::Scalar)transform.c2, (Athena::Scalar)transform.c3, (Athena::Scalar)transform.d3,
                                                    (Athena::Scalar)transform.a4, (Athena::Scalar)transform.b4, (Athena::Scalar)transform.c4, (Athena::Scalar)transform.d4);
                    
                    Athena::Matrix4::DecomposeMatrixInScaleRotateTranslateComponents(AthenaTransform, scale, rotation, position);
                }
                aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
                
                processMesh(mesh, scene, obj, position, scale, rotation);
            }

            obj->transform->parent = parent;
            if (parent != nullptr)
                parent->children.push_back(obj->transform);
        }
        
        for(GLuint i = 0; i < node->mNumChildren; ++i)
        {
            processMeshNode(node->mChildren[i], scene, nullptr);
        }
    }

    //process a node recursively and for each node processed, process his meshes
    void Model::processNode(aiNode* node, const aiScene* scene, const std::string& name, Transform* parent)
    {
        if(node == scene->mRootNode)
        {
            Odysseus::SceneObject* obj = new SceneObject(name);
            for(GLuint i = 0; i < node->mNumChildren; ++i)
            {
                processMeshNode(node->mChildren[i], scene, obj->transform);
            }
        }
    }

    void Model::setMeshTextures(aiMaterial* material, PhongMaterial& mat)
    {
        //diffuse
        if(material->GetTextureCount(aiTextureType_DIFFUSE) > 0)
        {
            this->_gammaCorrect = true;
            std::vector<Texture2D> diffuseMaps = loadTexture(material, aiTextureType_DIFFUSE, this->_gammaCorrect);
            mat.Textures.insert(mat.Textures.end(), diffuseMaps.begin(), diffuseMaps.end());
        }
        //specular
        if(material->GetTextureCount(aiTextureType_SPECULAR) > 0)
        {
            this->_gammaCorrect = false;
            std::vector<Texture2D> specularMaps = loadTexture(material, aiTextureType_SPECULAR, this->_gammaCorrect);
            mat.Textures.insert(mat.Textures.end(), specularMaps.begin(), specularMaps.end());
        }

        if(material->GetTextureCount(aiTextureType_NORMALS) > 0)
        {
            this->_gammaCorrect = false;
            std::vector<Texture2D> normalMap = loadTexture(material, aiTextureType_NORMALS, this->_gammaCorrect);
            mat.Textures.insert(mat.Textures.end(), normalMap.begin(), normalMap.end());
        }
    }

    void Model::setMeshMaterials(aiMaterial* material, PhongMaterial& mat)
    {
        aiColor3D diffuse, ambient, specular;
        float shininess;

        if(AI_SUCCESS == material->Get(AI_MATKEY_SHININESS, shininess))
            mat.Shininess = shininess;
        
        if(AI_SUCCESS == material->Get(AI_MATKEY_COLOR_DIFFUSE, diffuse))
            mat.Diffuse = Athena::Vector3(diffuse.r, diffuse.g, diffuse.b);
        
        if(AI_SUCCESS == material->Get(AI_MATKEY_COLOR_AMBIENT, ambient))
            mat.Ambient = Athena::Vector3(ambient.r, ambient.g, ambient.b);

        if(AI_SUCCESS == material->Get(AI_MATKEY_COLOR_SPECULAR, specular))
            mat.Specular = Athena::Vector3(specular.r, specular.g, specular.b);
    }

    void Model::setMeshPBRtextures(aiMaterial* material, PhysicsMaterial& mat)
    {
        if(material->GetTextureCount(aiTextureType_DIFFUSE) > 0)
        {
            std::cout << "Has texture albedo" << std::endl;
            this->_gammaCorrect = true;
            std::vector<Texture2D> albedoMap = loadTexture(material, aiTextureType_DIFFUSE, this->_gammaCorrect);
            mat.PBR_textures.insert(mat.PBR_textures.end(), albedoMap.begin(), albedoMap.end());
        } else
        {
            aiColor4D color;
            if(AI_SUCCESS == material->Get(AI_MATKEY_COLOR_DIFFUSE, color))
            {
                std::cout << "has material albedo" << std::endl;
                mat.albedo = Athena::Vector4(color.r, color.g, color.b, color.a);
            }
        }

        if(material->GetTextureCount(aiTextureType_NORMALS) > 0)
        {
            std::cout << "Has texture normal" << std::endl;
            this->_gammaCorrect = false;
            std::vector<Texture2D> normalMap = loadTexture(material, aiTextureType_NORMALS, this->_gammaCorrect);
            mat.PBR_textures.insert(mat.PBR_textures.end(), normalMap.begin(), normalMap.end());
        }

        if(material->GetTextureCount(aiTextureType_METALNESS) > 0)
        {
            std::cout << "Has texture metallic" << std::endl;
            this->_gammaCorrect = false;
            std::vector<Texture2D> metalnessMap = loadTexture(material, aiTextureType_METALNESS, this->_gammaCorrect);
            mat.PBR_textures.insert(mat.PBR_textures.end(), metalnessMap.begin(), metalnessMap.end());
        } 
        else
        {
            float metallic;
            if(AI_SUCCESS == material->Get(AI_MATKEY_METALLIC_FACTOR, metallic))
            {
                std::cout << "has material metallic" << std::endl;
                mat.metallic = metallic;
            }
        }

        if(material->GetTextureCount(aiTextureType_DIFFUSE_ROUGHNESS) > 0)
        {
            std::cout << "Has texture roughness" << std::endl;
            this->_gammaCorrect = false;
            std::vector<Texture2D> roughnessMap = loadTexture(material, aiTextureType_DIFFUSE_ROUGHNESS, this->_gammaCorrect);
            mat.PBR_textures.insert(mat.PBR_textures.end(), roughnessMap.begin(), roughnessMap.end());
        }
        else
        {
            float roughness;
            if(AI_SUCCESS == material->Get(AI_MATKEY_ROUGHNESS_FACTOR, roughness))
            {
                std::cout << "has material roughness" << std::endl;
                mat.roughness = roughness;
            }
        }

        if(material->GetTextureCount(aiTextureType_AMBIENT_OCCLUSION) > 0)
        {
            std::cout << "Has texture ao" << std::endl;
            this->_gammaCorrect = false;
            std::vector<Texture2D> AOMap = loadTexture(material, aiTextureType_AMBIENT_OCCLUSION, this->_gammaCorrect);
            mat.PBR_textures.insert(mat.PBR_textures.end(), AOMap.begin(), AOMap.end());
        }
    }
    
    void Model::processMesh(aiMesh *mesh, const aiScene *scene, SceneObject* obj, Athena::Vector3& position, Athena::Vector3& scale, Athena::Quaternion& rotation)
    {
        // data to fill
        Vertices vertices;
        std::vector<GLuint> indices;
        Athena::Vector3 avg;

        if(objectType == "obj")
        {
            for(GLuint i = 0; i < mesh->mNumVertices; ++i)
            {
                avg.coordinates.x += mesh->mVertices[i].x;
                avg.coordinates.y += mesh->mVertices[i].y;
                avg.coordinates.z += mesh->mVertices[i].z;
            }
        }

        // walk through each of the mesh's vertices
        for(GLuint i = 0; i < mesh->mNumVertices; ++i)
        {
            Athena::Vector3 vector;
            if(objectType == "obj")
            {
                // positions
                vector.coordinates.x = mesh->mVertices[i].x - (avg.coordinates.x / mesh->mNumVertices);
                vector.coordinates.y = mesh->mVertices[i].y - (avg.coordinates.y / mesh->mNumVertices);
                vector.coordinates.z = mesh->mVertices[i].z - (avg.coordinates.z / mesh->mNumVertices);
            }
            else
            {
            // positions
            vector.coordinates.x = mesh->mVertices[i].x;
            vector.coordinates.y = mesh->mVertices[i].y;
            vector.coordinates.z = mesh->mVertices[i].z;
            }
            vertices.Positions.push_back(vector);
            // normals
            if (mesh->HasNormals())
            {
                vector.coordinates.x = mesh->mNormals[i].x;
                vector.coordinates.y = mesh->mNormals[i].y;
                vector.coordinates.z = mesh->mNormals[i].z;
                vertices.Normals.push_back(vector);
            }
            // texture coordinates
            if(mesh->mTextureCoords[0]) // does the mesh contain texture coordinates?
            {
                Athena::Vector2 vec;
                // a vertex can contain up to 8 different texture coordinates. We thus make the assumption that we won't 
                // use models where a vertex can have multiple texture coordinates so we always take the first set (0).
                vec.coordinates.x = mesh->mTextureCoords[0][i].x; 
                vec.coordinates.y = mesh->mTextureCoords[0][i].y;
                vertices.TexCoords.push_back(vec);
                // tangent
                vector.coordinates.x = mesh->mTangents[i].x;
                vector.coordinates.y = mesh->mTangents[i].y;
                vector.coordinates.z = mesh->mTangents[i].z;
                vertices.Tangents.push_back(vector);
            }
            else{
                vertices.TexCoords.push_back(Athena::Vector2(0.0f, 0.0f));
            }
        }
        //Getting the indices from each mesh face
        for(GLuint i = 0; i < mesh->mNumFaces; ++i)
        {
            aiFace face = mesh->mFaces[i];
            for(GLuint j = 0; j < face.mNumIndices; ++j)
                indices.push_back(face.mIndices[j]);        
        } 

        //process materials
        aiMaterial* material = scene->mMaterials[mesh->mMaterialIndex];
        auto objMesh = obj->addComponent<Odysseus::Mesh>();

        if(objectType == "gltf")
        {
        //GLTF positions
            objMesh->transform->position = position;
            objMesh->transform->rotation = rotation.inverse();
            objMesh->transform->localScale = scale;
        }

        if(objectType == "fbx")
        {
        //FBX positions
            objMesh->transform->position.coordinates.x = position.coordinates.x / 100;
            objMesh->transform->position.coordinates.y = position.coordinates.y / 100;
            objMesh->transform->position.coordinates.z = position.coordinates.z / 100;
            objMesh->transform->rotation = rotation;
            objMesh->transform->localScale = scale / 100;
        }

        if(objectType == "obj")
        {
        //OBJ positions
        objMesh->transform->position.coordinates.x = avg.coordinates.x / mesh->mNumVertices;
        objMesh->transform->position.coordinates.y = avg.coordinates.y / mesh->mNumVertices;
        objMesh->transform->position.coordinates.z = avg.coordinates.z / mesh->mNumVertices;
        }
        
        objMesh->setVertices(vertices);
        objMesh->setIndices(indices);
        objMesh->setIfPBR(_isPBR);
        if(_isPBR)
        {
            PhysicsMaterial physMat;
            setMeshPBRtextures(material, physMat);
            if(physMat.PBR_textures.size() > 0)
                objMesh->setShader(this->textureShader);
            else
                objMesh->setShader(this->materialShader);
            objMesh->setPhysicsMaterial(physMat);
        }else
        {
            PhongMaterial phongMat;
            setMeshTextures(material, phongMat);
            setMeshMaterials(material, phongMat);
            objMesh->setPhongMaterial(phongMat);
        }
        
    }

    std::vector<Texture2D> Model::loadTexture(aiMaterial* mat, aiTextureType type, bool gammaCorrect)
    {
        std::vector<Texture2D> textures;
        for(GLuint i = 0; i < mat->GetTextureCount(type); ++i)
        {
            aiString str;
            mat->GetTexture(type, i, &str);
            //checking if texture was already loaded, if so skip loading it another time
            bool skip = false;
            for(GLuint j = 0; j < textures_loaded.size(); ++j)
            {
                if(std::strcmp(textures_loaded[j].path.data(), str.C_Str()) == 0)
                {
                    textures.push_back(textures_loaded[j]);
                    skip = true;
                    break;
                }
            }

            if(!skip)
            {
                Texture2D tex(this->directory, str.C_Str(), type);
                tex.loadTextureFromFile(gammaCorrect);
                textures.push_back(tex);
                textures_loaded.push_back(tex);
            }
        }
        return textures;
    }
}

