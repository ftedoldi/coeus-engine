#ifndef MODEL_HPP
#define MODEL_HPP

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <coeus.hpp>
#include <Component.hpp>
#include <Camera.hpp>
#include <Tree.hpp>
#include <PhysicsMaterial.hpp>

#include "Mesh.hpp"
#include "../Texture/Texture2D.hpp"

namespace Odysseus
{
    std::vector<Texture2D> loadTexture(aiMaterial *mat, aiTextureType type);
    class Model 
    {
    public:
        std::vector<Texture2D> textures_loaded;
        std::vector<Mesh> meshes;
        std::string directory;
        Odysseus::Shader* shader;

        //Deleting the possibility to use a copy constructor and copy operator=
        Model(const Model& model) = delete;
        Model& operator=(const Model& copy) = delete;

        //Creating a default move constructor and move operator=
        Model& operator=(Model&& move) noexcept = default;
        Model(Model&& model) = default;

        //Creating a default constructor
        Model(const std::string& path, Shader* shader);

        //rendering the model by drawing each Mesh instance in the vector
        void Draw(Shader* shader);

    private:

        void loadModel(const std::string& path);
        Material loadMaterial(aiMaterial* mat);
        std::vector<Texture2D> loadTexture(aiMaterial *mat, aiTextureType type);

        void setMeshTextures(aiMaterial* material, Material& mat);
        void setMeshMaterials(aiMaterial* material, Material& mat);

        void setMeshPBRtextures(aiMaterial* material, PhysicsMaterial& mat);
        void setMeshPBRmaterial(aiMaterial* material, PhysicsMaterial& mat);
        void processMesh(aiMesh* mesh, const aiScene* scene, SceneObject* sceneObject);
        void processNode(aiNode* node, const aiScene* scene);   

    };
}

#endif