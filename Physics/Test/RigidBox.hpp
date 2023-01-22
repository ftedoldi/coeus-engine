#ifndef RIGIDBOX_HPP
#define RIGIDBOX_HPP
#include <RigidPhysicsEngine.hpp>
#include <Shader.hpp>
#include <SerializableClass.hpp>

namespace Odysseus
{
    class Mesh;
    class SceneObject;
}

namespace System
{
    class Component;
}

struct BBox
{
    Athena::Vector3 size;
};

namespace Khronos
{

    class RigidBox : public System::Component
    {
        //class Mesh;
        RigidBody* rigidBody;
        RigidPhysicsEngine* physicSimulation;
        CollisionBox* cBox;
        Odysseus::Shader* bboxShader;
        Odysseus::Mesh* meshComponent;
        BBox boundingBox;

    public:

        RigidBox();

        double mass = 1.0;
        double damping = 0.9;

        virtual void start();
        virtual void update();

        virtual void setOrderOfExecution(const short& newOrderOfExecution);

        virtual int getUniqueID();

        virtual std::string toString();

        virtual void startRuntime();
        virtual void updateRuntime();

        void createBBox();
        // void setupBox();
        // void drawBox();
        void setPhysicsSimulation(RigidPhysicsEngine* physSim);

        virtual void serialize(YAML::Emitter& out);
        virtual System::Component* deserialize(YAML::Node& node);

        ~RigidBox();

        SERIALIZABLE_CLASS(System::Component);
    };
}

#endif