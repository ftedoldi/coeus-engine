#ifndef MODEL_HPP
#define MODEL_HPP

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <coeus.hpp>
#include <Component.hpp>
#include <Camera.hpp>

#include "Mesh.hpp"
#include "../Texture/Texture2D.hpp"

namespace Odysseus
{
    std::vector<Texture2D> loadTexture(aiMaterial *mat, aiTextureType type);
    class Model : public System::Component
    {
    public:
        std::vector<Texture2D> textures_loaded;
        std::vector<Mesh> meshes;
        std::string directory;
        const std::string& path;
        Odysseus::Shader* shader;
        Odysseus::Camera* camera;

        //Deleting the possibility to use a copy constructor and copy operator=
        Model(const Model& model) = delete;
        Model& operator=(const Model& copy) = delete;

        //Creating a default move constructor and move operator=
        Model& operator=(Model&& move) noexcept = default;
        Model(Model&& model) = default;

        //Creating a default constructor
        Model();

        //rendering the model by drawing each Mesh instance in the vector
        void Draw(Shader* shader);

        void setPath(const std::string& path);
        void setShader(Shader* shader);
        void setCamera(Camera* camera);

        virtual void start();
        virtual void update();

        virtual void setOrderOfExecution(const short& newOrderOfExecution);

        virtual short getUniqueID();

        virtual std::string toString();

    private:
        //Loading the model using assimp library, passing by copy the path, for the needs to be modified
        std::string _path;

        void loadModel(const std::string& path);
        Material loadMaterial(aiMaterial* mat);
        std::vector<Texture2D> loadTexture(aiMaterial *mat, aiTextureType type);

        void setMeshTextures(aiMaterial* material, Material& mat);
        void setMeshMaterials(aiMaterial* material, Material& mat);
        Mesh processMesh(aiMesh* mesh, const aiScene* scene);
        void processNode(aiNode* node, const aiScene* scene);   

    };
}

#endif