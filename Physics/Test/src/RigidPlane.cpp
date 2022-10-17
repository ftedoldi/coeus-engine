#include "../RigidPlane.hpp"
#include <Mesh.hpp>
#include <imgui.h>
#include <imgui_internal.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <Texture2D.hpp>
#include <Folder.hpp>

namespace Khronos
{

    RigidPlane::RigidPlane()
    {
        this->direction = Athena::Vector3(0.0f, 1.0f, 0.0f);
    }

    RigidPlane::RigidPlane(Athena::Vector3& dir, Athena::Scalar off)
    {
        this->direction = dir;
        this->offset = off;
    }

    RigidPlane::~RigidPlane()
    {

    }

    void RigidPlane::start()
    {
        this->cPlane = new CollisionPlane(this->direction.normalized(), this->sceneObject->transform->position.coordinates.y);
    }

    void RigidPlane::update()
    {

    }

    void RigidPlane::startRuntime()
    {
        this->physicSimulation->instance->collisionGenerator->planes.push_back(this->cPlane);
    }

    void RigidPlane::updateRuntime()
    {

    }

    void RigidPlane::setOrderOfExecution(const short& newOrderOfExecution)
    {

    }

    int RigidPlane::getUniqueID()
    {
        return this->_uniqueID;
    }

    std::string RigidPlane::toString()
    {
        return "RigidPlane";
    }

    void RigidPlane::setPhysicsSimulation(RigidPhysicsEngine* physSim)
    {
        this->physicSimulation = physSim;
    }

    void RigidPlane::showComponentFieldsInEditor()
    {
        ImGui::Text("Plane direction:");
        float planeDir[] = { 
                            this->direction.coordinates.x, 
                            this->direction.coordinates.y, 
                            this->direction.coordinates.z 
                        };
        ImGui::InputFloat3("Plane Direction", planeDir);
        this->direction = Athena::Vector3(planeDir);
        this->cPlane->direction = this->direction;
    }

    void RigidPlane::serialize(YAML::Emitter& out)
    {
        out << YAML::Key << this->toString();
        out << YAML::BeginMap;
            out << YAML::Key << "Direction";
            out << YAML::BeginMap;
                out << YAML::Key << "X" << YAML::Value << this->direction.coordinates.x;
                out << YAML::Key << "Y" << YAML::Value << this->direction.coordinates.y;
                out << YAML::Key << "Z" << YAML::Value << this->direction.coordinates.z;
            out << YAML::EndMap;
            out << YAML::Key << "Offset" << YAML::Value << this->offset;
        out << YAML::EndMap;
    }

    System::Component* RigidPlane::deserialize(YAML::Node& node)
    {
        auto component = node[this->toString()];

        this->direction = Athena::Vector3();

        this->direction.coordinates.x = component["Direction"]["X"].as<Athena::Scalar>();
        this->direction.coordinates.y = component["Direction"]["Y"].as<Athena::Scalar>();
        this->direction.coordinates.z = component["Direction"]["Z"].as<Athena::Scalar>();

        this->offset = component["Offset"].as<Athena::Scalar>();

        return this;
    }

    SERIALIZE_CLASS
    {
        System::SerializableClass::registerClass<RigidPlane>("RigidPlane");
    }
}