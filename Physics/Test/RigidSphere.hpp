#ifndef RIGIDSPHERE_HPP
#define RIGIDSPHERE_HPP
#include <RigidPhysicsEngine.hpp>
#include <Shader.hpp>
#include <SerializableClass.hpp>

namespace Khronos
{
    struct BSphere
    {
        Athena::Scalar radius;
    };

    class RigidSphere : public System::Component
    {
        RigidBody* rigidBody;
        RigidPhysicsEngine* physicSimulation;
        CollisionSphere* cSphere;
        Odysseus::Mesh* meshComponent;
        BSphere boundingSphere;
        
    public:
        RigidSphere();

        double mass = 1.0;
        double damping = 0.9;

        virtual void start();
        virtual void update();

        virtual void setOrderOfExecution(const short& newOrderOfExecution);

        virtual int getUniqueID();

        virtual std::string toString();

        virtual void startRuntime();
        virtual void updateRuntime();

        void createBSphere();
        void setPhysicsSimulation(RigidPhysicsEngine* physSim);

        virtual void serialize(YAML::Emitter& out);
        virtual System::Component* deserialize(YAML::Node& node);

        virtual ~RigidSphere();

        SERIALIZABLE_CLASS(System::Component);

    };
}

#endif