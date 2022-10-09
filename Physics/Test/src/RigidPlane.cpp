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
        //this->_uniqueID = 4121;
        //if(this->transform == nullptr)
            //this->transform->position.print();
        //cPlane = new CollisionPlane(Athena::Vector3::up(), this->transform->position.coordinates.y);
        //this->_hasEditorTexture = false;
    }

    RigidPlane::~RigidPlane()
    {
        //delete this->cPlane;
    }

    void RigidPlane::start()
    {
        this->cPlane = new CollisionPlane(Athena::Vector3::up(), this->sceneObject->transform->position.coordinates.y);
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

    /*void RigidPlane::serialize(YAML::Emitter& out)
    {
        out << YAML::Key << this->toString();
        out << YAML::BeginMap;
            out << YAML::Key << NAMEOF(asd) << YAML::Value << this->asd;
            out << YAML::Key << NAMEOF(var) << YAML::Value << this->var;
        out << YAML::EndMap; 
    }
    System::Component* RigidPlane::deserialize(YAML::Node& node)
    {

    }*/

    void RigidPlane::showComponentFieldsInEditor()
    {
        ImGui::InputFloat(NAMEOF("Offset"), &this->cPlane->offset);
        ImGui::Text("Plane direction:");
            float planeDir[] = { 
                                this->cPlane->direction.coordinates.x, 
                                this->cPlane->direction.coordinates.y, 
                                this->cPlane->direction.coordinates.z 
                            };
            ImGui::InputFloat3("Plane Direction", planeDir);
    }

    void RigidPlane::serialize(YAML::Emitter& out)
    {
        /*out << YAML::Key << this->toString();
        out << YAML::BeginMap;
            out << YAML::Key << "Bounding box";
            out << YAML::Key << "Size";
            out << YAML::BeginMap;
                out << YAML::Key << "X" << YAML::Value << this->boundingBox.size.coordinates.x;
                out << YAML::Key << "Y" << YAML::Value << this->boundingBox.size.coordinates.y;
                out << YAML::Key << "Z" << YAML::Value << this->boundingBox.size.coordinates.z;
            out << YAML::EndMap;*/
    }

    System::Component* RigidPlane::deserialize(YAML::Node& node)
    {
        return this;
    }

    SERIALIZE_CLASS
    {
        System::SerializableClass::registerClass<RigidPlane>("RigidPlane");
    }
}