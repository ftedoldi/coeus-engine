#ifndef MODEL_HPP
#define MODEL_HPP

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <assimp/version.h>
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
        std::vector<SceneObject*> objectsCreated;
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

    private:

        bool _gammaCorrect;
        void loadModel(const std::string& path);
        std::vector<Texture2D> loadTexture(aiMaterial *mat, aiTextureType type, bool gammaCorrect);

        void setMeshTextures(aiMaterial* material, PhongMaterial& mat);
        void setMeshMaterials(aiMaterial* material, PhongMaterial& mat);

        void setMeshPBRtextures(aiMaterial* material, PhysicsMaterial& mat);
        void setMeshPBRmaterial(aiMaterial* material, PhysicsMaterial& mat);
        void processMesh(aiMesh* mesh, const aiScene* scene, SceneObject* sceneObject);
        void processNode(aiNode* node, const aiScene* scene, Transform* parent=nullptr);   

    };
}

#endif