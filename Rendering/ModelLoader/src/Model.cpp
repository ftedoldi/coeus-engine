#include "../Model.hpp"
namespace Odysseus
{

    Model::Model(const std::string& path, Shader* shader) : shader(shader)
    {
        loadModel(path);
    }

    void Model::loadModel(const std::string& path)
    {
        Assimp::Importer importer;
        const aiScene* scene = importer.ReadFile(path, aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_CalcTangentSpace /*| aiProcess_FlipUVs*/);
        //checking for errors in the scene creation
        if(!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
        {
            std::cout << "ASSIMP SCENE ERROR: " << importer.GetErrorString() << std::endl;
            return;
        }
        this->directory = path.substr(0, path.find_last_of('/'));
        std::cout << "Path at loadModel: " << path << std::endl;
        std::cout << "Directory at loadModel: " << directory << std::endl;

        processNode(scene->mRootNode, scene);

    }

    //process a node recursively and for each node processed, process his meshes
    void Model::processNode(aiNode* node, const aiScene* scene)
    {
        for(GLuint i = 0; i < node->mNumMeshes; ++i)
        {
            Odysseus::SceneObject* obj = new SceneObject();
            aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
            processMesh(mesh, scene, obj);
        }

        for(GLuint i = 0; i < node->mNumChildren; ++i)
        {
            processNode(node->mChildren[i], scene);
        }
    }

    void Model::setMeshTextures(aiMaterial* material, Material& mat)
    {
        //diffuse
        if(material->GetTextureCount(aiTextureType_DIFFUSE) > 0)
        {
            std::vector<Texture2D> diffuseMaps = loadTexture(material, aiTextureType_DIFFUSE);
            mat.Textures.insert(mat.Textures.end(), diffuseMaps.begin(), diffuseMaps.end());
        }
        //specular
        if(material->GetTextureCount(aiTextureType_SPECULAR) > 0)
        {
            std::vector<Texture2D> specularMaps = loadTexture(material, aiTextureType_SPECULAR);
            std::cout << "Has texture specular" << std::endl;
            mat.Textures.insert(mat.Textures.end(), specularMaps.begin(), specularMaps.end());
        }
        //normal
        if(material->GetTextureCount(aiTextureType_HEIGHT) > 0)
        {
            std::vector<Texture2D> normalMaps = loadTexture(material, aiTextureType_HEIGHT);
            std::cout << "Has texture height" << std::endl;
            mat.Textures.insert(mat.Textures.end(), normalMaps.begin(), normalMaps.end());
        }
        //height
        if(material->GetTextureCount(aiTextureType_AMBIENT) > 0)
        {
            std::vector<Texture2D> heightMaps = loadTexture(material, aiTextureType_AMBIENT);
            std::cout << "Has texture ambient" << std::endl;
            mat.Textures.insert(mat.Textures.end(), heightMaps.begin(), heightMaps.end());
        }

        if(material->GetTextureCount(aiTextureType_SHININESS) > 0)
        {
            std::vector<Texture2D> shininessMaps = loadTexture(material, aiTextureType_SHININESS);
            std::cout << "Has texture shininess" << std::endl;
            mat.Textures.insert(mat.Textures.end(), shininessMaps.begin(), shininessMaps.end());
            
        }
        std::cout <<"Textures size: "<< mat.Textures.size() << std::endl;
    }

    void Model::setMeshMaterials(aiMaterial* material, Material& mat)
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

    void Model::processMesh(aiMesh *mesh, const aiScene *scene, SceneObject* obj)
    {
        // data to fill
        std::vector<Vertex> vertices;
        std::vector<GLuint> indices;

        // walk through each of the mesh's vertices
        for(GLuint i = 0; i < mesh->mNumVertices; ++i)
        {
            Vertex vertex;
            Athena::Vector3 vector;
            // positions
            vector.coordinates.x = mesh->mVertices[i].x;
            vector.coordinates.y = mesh->mVertices[i].y;
            vector.coordinates.z = mesh->mVertices[i].z;
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
                // bitangent
                vector.coordinates.x = mesh->mBitangents[i].x;
                vector.coordinates.y = mesh->mBitangents[i].y;
                vector.coordinates.z = mesh->mBitangents[i].z;
                vertex.Bitangent = vector;
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
        Material mat;
        setMeshTextures(material, mat);
        setMeshMaterials(material, mat);

        auto objMesh = obj->addComponent<Odysseus::Mesh>();
        objMesh->setShader(this->shader);
        objMesh->setVertices(vertices);
        objMesh->setIndices(indices);
        objMesh->setMaterial(mat);

    }

    std::vector<Texture2D> Model::loadTexture(aiMaterial* mat, aiTextureType type)
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
                tex.loadTextureFromFile();
                textures.push_back(tex);
                textures_loaded.push_back(tex);
            }
        }
        return textures;
    }
}

