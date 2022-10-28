#ifndef MESH_HPP
#define MESH_HPP

#include <Component.hpp>

#include <SerializableClass.hpp>

#include <coeus.hpp>
#include <Shader.hpp>
#include "../Texture/Texture2D.hpp"
#include <PhongMaterial.hpp>
#include <PhysicsMaterial.hpp>
#include <Time.hpp>

#include <PickableObject.hpp>
#include <RigidPhysicsEngine.hpp>
#include <CUDAInteroperationManager.hpp>

#include <vector>
#include <random>

namespace CUDA::Interop
{
    class CUDAInteroperationManager;
}

namespace Odysseus
{
    class EditorCamera;

    struct Vertices
    {
        std::vector<Athena::Vector3> Positions;
        std::vector<Athena::Vector3> Normals;
        std::vector<Athena::Vector2> TexCoords;
        std::vector<Athena::Vector3> Tangents;
    };

    //TODO: change the location of it
    struct Particle
    {
        //Vertex vertex;
        Athena::Vector3 position;
        Athena::Vector3 positionPredicted;
        Athena::Vector3 velocity;
        Athena::Scalar inverseMass;
    };

    struct Edge
    {
        Particle particles[2];
    };

    struct Triangle
    {
        Edge edges[3];
    };

    struct TriangleAdj
    {
        Triangle triangle;
        std::vector<Triangle*> triangleAdjs;
    };

    class Mesh : public System::Component
    {
    public:
        Vertices vertices;
        std::vector<GLuint> indices;
        PhongMaterial phongMaterial;
        PhysicsMaterial physicsMaterial;
        CUDA::Interop::CUDAInteroperationManager* cudaInterop;
        cudaGraphicsResource *cuda_vbo_resource;

            std::string path;

            Shader *shader;
            GLuint VAO;

            Mesh();
            ~Mesh() noexcept;

            void setVertices(Vertices& vertices);
            void setIndices(std::vector<GLuint> &indices);
            void setPhongMaterial(PhongMaterial &mat);
            void setPhysicsMaterial(PhysicsMaterial &mat);
            void setShader(Shader *shader);
            void setIfPBR(bool isPBR);
            void setupRigidBody();

            bool operator == (const Mesh& m) const;

            virtual void start();
            virtual void update();

            virtual void setOrderOfExecution(const short &newOrderOfExecution);

            virtual int getUniqueID();

            virtual std::string toString();

            virtual void showComponentFieldsInEditor();

            virtual void serialize(YAML::Emitter& out);
            virtual System::Component* deserialize(YAML::Node& node);

            SERIALIZABLE_CLASS(System::Component);

    private:
        GLuint VBO, EBO;

        bool hasTexture;
        bool _isPBR;

        float _uniqueFloatID;

        // Inizialize VAO, VBO, EBO
        void setupMesh();
        void updateMeshComponent();

        void freeGPUresources();
    };
}
#endif