#include "../RigidSphere.hpp"
#include <Mesh.hpp>
#include <SceneObject.hpp>
#include <Component.hpp>
#include <Texture2D.hpp>
#include <Folder.hpp>

namespace Khronos
{
    RigidSphere::RigidSphere()
    {
        this->_editorTextureID = Odysseus::Texture2D::loadTextureFromFile(
                                                                            (System::Folder::getFolderPath("Icons").string() + "/BSphere.png").c_str(), 
                                                                            true
                                                                        ).ID;
        this->_hasEditorTexture = true;
    }

    RigidSphere::~RigidSphere()
    {

    }

    void RigidSphere::start()
    {
        for (int i = 0; i < this->sceneObject->_container->components.size(); i++)
        {
            if (this->sceneObject->_container->components[i]->toString() == "Mesh")
            {
                std::cout << "has component mesh" << std::endl;
                this->meshComponent = dynamic_cast<Odysseus::Mesh*>(this->sceneObject->_container->components[i]);
            } 
        }

        createBSphere();
        auto radius = this->boundingSphere.radius;
        Athena::Matrix3 it;

        this->rigidBody = new RigidBody();
        this->rigidBody->setPosition(this->sceneObject->transform->position);
        this->rigidBody->setOrientation(this->sceneObject->transform->rotation);
        this->rigidBody->setVelocity(Athena::Vector3(0.0, 0.0, 0.0));
        this->rigidBody->setRotation(Athena::Vector3(0.0, 0.0, 0.0));
        this->rigidBody->setMass(this->mass);
        this->rigidBody->setDamping(this->damping, this->damping);
        this->rigidBody->setInertiaTensor(it);
        this->rigidBody->setAwake(true);
        this->rigidBody->setCanSleep(false);
        this->rigidBody->sleepEpsilon = 0.3;

        this->cSphere = new CollisionSphere(radius);
        cSphere->body = this->rigidBody;

    }

    void RigidSphere::update()
    {

    }

    void RigidSphere::setOrderOfExecution(const short& newOrderOfExecution)
    {

    }

    int RigidSphere::getUniqueID()
    {
        return this->_uniqueID;
    }

    std::string RigidSphere::toString()
    {
        return "RigidSphere";
    }

    void RigidSphere::startRuntime()
    {
        this->physicSimulation->instance->bodyList.push_back(this->rigidBody);
        this->physicSimulation->instance->collisionGenerator->spheres.push_back(this->cSphere);
    }

    void RigidSphere::updateRuntime()
    {
        this->sceneObject->transform->position = this->rigidBody->position;
        this->sceneObject->transform->rotation = this->rigidBody->orientation;
    }

    void RigidSphere::createBSphere()
    {
        auto localScale = this->sceneObject->transform->localScale;

        Athena::Scalar min_x, max_x, min_y, max_y, min_z, max_z;
        min_x = max_x = meshComponent->vertices.at(0).Position.coordinates.x;
        min_y = max_y = meshComponent->vertices.at(0).Position.coordinates.y;
        min_z = max_z = meshComponent->vertices.at(0).Position.coordinates.z;

        for(int i = 1; i < meshComponent->vertices.size(); ++i)
        {
            if(meshComponent->vertices.at(i).Position.coordinates.x < min_x) min_x = meshComponent->vertices.at(i).Position.coordinates.x;
            if(meshComponent->vertices.at(i).Position.coordinates.y < min_y) min_y = meshComponent->vertices.at(i).Position.coordinates.y;
            if(meshComponent->vertices.at(i).Position.coordinates.z < min_z) min_z = meshComponent->vertices.at(i).Position.coordinates.z;

            if(meshComponent->vertices.at(i).Position.coordinates.x > max_x) max_x = meshComponent->vertices.at(i).Position.coordinates.x;
            if(meshComponent->vertices.at(i).Position.coordinates.y > max_y) max_y = meshComponent->vertices.at(i).Position.coordinates.y;
            if(meshComponent->vertices.at(i).Position.coordinates.z > max_z) max_z = meshComponent->vertices.at(i).Position.coordinates.z;
        }

        auto scaleX = this->sceneObject->transform->localScale.coordinates.x;
        auto scaleY = this->sceneObject->transform->localScale.coordinates.y;
        auto scaleZ = this->sceneObject->transform->localScale.coordinates.z;
        
        Athena::Scalar max = std::max({Athena::Math::scalarAbs(min_x * scaleX), Athena::Math::scalarAbs(min_y * scaleY), Athena::Math::scalarAbs(min_z * scaleZ),
                                       Athena::Math::scalarAbs(max_x * scaleX), Athena::Math::scalarAbs(max_y * scaleY), Athena::Math::scalarAbs(max_z * scaleZ)});

        this->boundingSphere.radius = max;
    }

    void RigidSphere::setPhysicsSimulation(RigidPhysicsEngine* physSim)
    {
        this->physicSimulation = physSim;
    }

    void RigidSphere::serialize(YAML::Emitter& out)
    {

    }

    System::Component* RigidSphere::deserialize(YAML::Node& node)
    {
        return this;
    }

    SERIALIZE_CLASS
    {
        System::SerializableClass::registerClass<RigidSphere>("RigidSphere");
    }

} // namespace Khronos
