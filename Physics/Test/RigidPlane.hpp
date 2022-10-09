#ifndef RIGIDPLANE_HPP
#define RIGIDPLANE_HPP
#include <CollisionGenerator.hpp>
#include <Component.hpp>
#include <SerializableClass.hpp>
#include <RigidPhysicsEngine.hpp>

namespace Odysseus
{
    class Mesh;
}

namespace Khronos
{
    
    class RigidPlane : public System::Component
    {
        
        RigidPhysicsEngine* physicSimulation;

    public:
        CollisionPlane* cPlane;
        RigidPlane();

        virtual void start();
        virtual void update();

        virtual void startRuntime();
        virtual void updateRuntime();

        virtual void setOrderOfExecution(const short& newOrderOfExecution);

        virtual int getUniqueID();

        virtual std::string toString();

        virtual ~RigidPlane();

        void setPhysicsSimulation(RigidPhysicsEngine* physSim);

        virtual void showComponentFieldsInEditor();

        virtual void serialize(YAML::Emitter& out);
        virtual System::Component* deserialize(YAML::Node& node);

        SERIALIZABLE_CLASS(System::Component);

    };
}

#endif