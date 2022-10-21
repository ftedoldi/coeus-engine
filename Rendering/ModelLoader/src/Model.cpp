#include "../Model.hpp"

#include <ModelBase.hpp>

namespace Odysseus
{

    Model::Model(const std::string& path, Shader* shader, bool isPBR) : shader(shader)
    {
        this->_isPBR = isPBR;
        loadModel(path);
    }

    // TODO: Set the position of a model in the world correctly
    void Model::loadModel(const std::string& path)
    {
        Assimp::Importer importer;
        const aiScene* scene = importer.ReadFile(path, aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_JoinIdenticalVertices | aiProcess_CalcTangentSpace | aiProcess_FlipUVs);
        //checking for errors in the scene creation
        if(!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
        {
            std::cout << "ASSIMP SCENE ERROR: " << importer.GetErrorString() << std::endl;
            return;
        }

        this->directory = path.substr(0, path.find_last_of('\\'));

        auto name = path.substr(path.find_last_of('\\') + 1);
        name = name.substr(0, name.find_last_of('.'));

        processNode(scene->mRootNode, scene);
    }

    //process a node recursively and for each node processed, process his meshes
    void Model::processNode(aiNode* node, const aiScene* scene, Transform* parent)
    {
        Odysseus::SceneObject* obj = new SceneObject();
        objectsCreated.push_back(obj);

        for(GLuint i = 0; i < node->mNumMeshes; ++i)
        {
            Athena::Vector3 position;
            if(node->mParent != nullptr)
            {
                //local transformation matrix of the node
                aiMatrix4x4 transform =  node->mTransformation;
                position = Athena::Vector3((float)transform.a4, (float)transform.b4, (float)transform.c4);
            }
            aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
            
            processMesh(mesh, scene, obj, position);
        }

        obj->transform->parent = parent;
        if (parent != nullptr)
            parent->children.push_back(obj->transform);

        // for(GLuint i = 0; i < node->mNumMeshes; ++i)
        // {
        //     Odysseus::SceneObject* obj = new SceneObject();
        //     aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
        //     processMesh(mesh, scene, obj);
        // }

        for(GLuint i = 0; i < node->mNumChildren; ++i)
        {
            processNode(node->mChildren[i], scene, obj->transform);
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
            this->_gammaCorrect = true;
            std::vector<Texture2D> albedoMap = loadTexture(material, aiTextureType_DIFFUSE, this->_gammaCorrect);
            mat.PBR_textures.insert(mat.PBR_textures.end(), albedoMap.begin(), albedoMap.end());
        }

        if(material->GetTextureCount(aiTextureType_NORMALS) > 0)
        {
            this->_gammaCorrect = false;
            std::vector<Texture2D> normalMap = loadTexture(material, aiTextureType_NORMALS, this->_gammaCorrect);
            mat.PBR_textures.insert(mat.PBR_textures.end(), normalMap.begin(), normalMap.end());
        }

        if(material->GetTextureCount(aiTextureType_METALNESS) > 0)
        {
            this->_gammaCorrect = false;
            std::vector<Texture2D> metalnessMap = loadTexture(material, aiTextureType_METALNESS, this->_gammaCorrect);
            mat.PBR_textures.insert(mat.PBR_textures.end(), metalnessMap.begin(), metalnessMap.end());
        }

        if(material->GetTextureCount(aiTextureType_DIFFUSE_ROUGHNESS) > 0)
        {
            this->_gammaCorrect = false;
            std::vector<Texture2D> roughnessMap = loadTexture(material, aiTextureType_DIFFUSE_ROUGHNESS, this->_gammaCorrect);
            mat.PBR_textures.insert(mat.PBR_textures.end(), roughnessMap.begin(), roughnessMap.end());
        }

        if(material->GetTextureCount(aiTextureType_AMBIENT_OCCLUSION) > 0)
        {
            this->_gammaCorrect = false;
            std::vector<Texture2D> AOMap = loadTexture(material, aiTextureType_AMBIENT_OCCLUSION, this->_gammaCorrect);
            mat.PBR_textures.insert(mat.PBR_textures.end(), AOMap.begin(), AOMap.end());
        }
    }

    void Model::setMeshPBRmaterial(aiMaterial* material, PhysicsMaterial& mat)
    {
        aiColor3D color;
        float metallic;
        float roughness;

        if(AI_SUCCESS == material->Get(AI_MATKEY_COLOR_DIFFUSE, color))
        {
            mat.albedo = Athena::Vector3(color.r, color.g, color.b);
        }
        
        if(AI_SUCCESS == material->Get(AI_MATKEY_METALLIC_FACTOR, metallic))
        {
            mat.metallic = metallic;
        }
            
        if(AI_SUCCESS == material->Get(AI_MATKEY_ROUGHNESS_FACTOR, roughness))
        {
            mat.roughness = roughness;
        }
    }

    void Model::processMesh(aiMesh *mesh, const aiScene *scene, SceneObject* obj, Athena::Vector3& position)
    {
        // data to fill
        std::vector<Vertex> vertices;
        std::vector<GLuint> indices;
        //Athena::Vector3 avg;

        // walk through each of the mesh's vertices
        for(GLuint i = 0; i < mesh->mNumVertices; ++i)
        {
            Vertex vertex;
            Athena::Vector3 vector;
            // positions
            vector.coordinates.x = mesh->mVertices[i].x;
            vector.coordinates.y = mesh->mVertices[i].y;
            vector.coordinates.z = mesh->mVertices[i].z;

            //Get average vertex position
            //avg.coordinates.x += mesh->mVertices[i].x;
            //avg.coordinates.y += mesh->mVertices[i].y;
            //avg.coordinates.z += mesh->mVertices[i].z;
            vertex.Position = vector;
            // normals
            if (mesh->HasNormals())
            {
                vector.coordinates.x = mesh->mNormals[i].x;
                vector.coordinates.y = mesh->mNormals[i].y;
                vector.coordinates.z = mesh->mNormals[i].z;
                vertex.Normal = vector;
            }
            // texture coordinates
            if(mesh->mTextureCoords[0]) // does the mesh contain texture coordinates?
            {
                Athena::Vector2 vec;
                // a vertex can contain up to 8 different texture coordinates. We thus make the assumption that we won't 
                // use models where a vertex can have multiple texture coordinates so we always take the first set (0).
                vec.coordinates.x = mesh->mTextureCoords[0][i].x; 
                vec.coordinates.y = mesh->mTextureCoords[0][i].y;
                vertex.TexCoords = vec;
                // tangent
                vector.coordinates.x = mesh->mTangents[i].x;
                vector.coordinates.y = mesh->mTangents[i].y;
                vector.coordinates.z = mesh->mTangents[i].z;
                vertex.Tangent = vector;
            }
            else{
                vertex.TexCoords = Athena::Vector2(0.0f, 0.0f);
            }
            vertices.push_back(vertex);
        }
        //Getting the indices from each mesh face
        for(GLuint i = 0; i < mesh->mNumFaces; ++i)
        {
            aiFace face = mesh->mFaces[i];
            for(GLuint j = 0; j < face.mNumIndices; ++j)
                indices.push_back(face.mIndices[j]);        
        } 

        //in the shaders, each texture must be named as 'texturetypeN' where N is a number ranging from 1 to the maximum number of the type of texture considered
        //and texturetype is the type of the texture e.g. diffuse, specular
        //for example, multiple diffuse textures will be written as diffuse1, diffuse2, diffuse3, ecc.

        //process materials
        aiMaterial* material = scene->mMaterials[mesh->mMaterialIndex];
        auto objMesh = obj->addComponent<Odysseus::Mesh>();

        //GLTF positions
        //objMesh->transform->position.coordinates.x = position.coordinates.x;
        //objMesh->transform->position.coordinates.y = position.coordinates.y;
        //objMesh->transform->position.coordinates.z = position.coordinates.z;

        //FBX positions
        //objMesh->transform->position.coordinates.x = position.coordinates.x / 100;
        //objMesh->transform->position.coordinates.y = position.coordinates.y / 100;
        //objMesh->transform->position.coordinates.z = position.coordinates.z / 100;

        //OBJ positions
        //objMesh->transform->position.coordinates.x = avg.coordinates.x / mesh->mNumVertices;
        //objMesh->transform->position.coordinates.y = avg.coordinates.y / mesh->mNumVertices;
        //objMesh->transform->position.coordinates.z = avg.coordinates.z / mesh->mNumVertices;
        objMesh->setShader(this->shader);
        objMesh->setVertices(vertices);
        objMesh->setIndices(indices);
        objMesh->setIfPBR(_isPBR);
        if(_isPBR)
        {
            PhysicsMaterial physMat;
            setMeshPBRtextures(material, physMat);
            setMeshPBRmaterial(material, physMat);
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

