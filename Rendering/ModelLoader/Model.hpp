#ifndef MODEL_HPP
#define MODEL_HPP

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <assimp/version.h>
#include <coeus.hpp>
#include <EditorCamera.hpp>
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
        Odysseus::Shader* textureShader;
        Odysseus::Shader* materialShader;

        //Deleting the possibility to use a copy constructor and copy operator=
        Model(const Model& model) = delete;
        Model& operator=(const Model& copy) = delete;

        //Creating a default move constructor and move operator=
        Model& operator=(Model&& move) noexcept = default;
        Model(Model&& model) = default;

        //Creating a default constructor
        Model(const std::string& path, Shader* textureShader, Shader* materialShader, bool isPBR);

    private:

        bool _gammaCorrect;
        bool _isPBR;
        void loadModel(const std::string& path);
        std::vector<Texture2D> loadTexture(aiMaterial *mat, aiTextureType type, bool gammaCorrect);

        void setMeshTextures(aiMaterial* material, PhongMaterial& mat);
        void setMeshMaterials(aiMaterial* material, PhongMaterial& mat);

        void setMeshPBRtextures(aiMaterial* material, PhysicsMaterial& mat);
        void processMesh(aiMesh* mesh, const aiScene* scene, SceneObject* sceneObject, Athena::Vector3& transform);
        void processNode(aiNode* node, const aiScene* scene, Transform* parent=nullptr);
        void processTransformations(aiNode* node, aiMatrix4x4& transformation); 

    };
}

#endif