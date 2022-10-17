#include "../RigidBox.hpp"
#include <Mesh.hpp>
#include <SceneObject.hpp>
#include <Component.hpp>
#include <Texture2D.hpp>
#include <Folder.hpp>

namespace Khronos
{
    RigidBox::RigidBox()
    {
        this->_editorTextureID = Odysseus::Texture2D::loadTextureFromFile(
                                                                            (System::Folder::getFolderPath("Icons").string() + "/BBox.png").c_str(), 
                                                                            true
                                                                        ).ID;
        this->_hasEditorTexture = true;
    }

    RigidBox::~RigidBox()
    {

    }

    void RigidBox::start()
    {

        for (int i = 0; i < this->sceneObject->_container->components.size(); i++)
        {
            if (this->sceneObject->_container->components[i]->toString() == "Mesh")
            {
                std::cout << "has component mesh" << std::endl;
                this->meshComponent = dynamic_cast<Odysseus::Mesh*>(this->sceneObject->_container->components[i]);
            } 
        }

        createBBox();
        auto scale = this->boundingBox.size;
        Athena::Matrix3 it;

        this->rigidBody = new RigidBody();
        this->rigidBody->setPosition(this->sceneObject->transform->position);
        this->rigidBody->setOrientation(this->sceneObject->transform->rotation);
        this->rigidBody->setVelocity(Athena::Vector3(0.0, 0.0, 0.0));
        this->rigidBody->setRotation(Athena::Vector3(0.0, 0.0, 0.0));
        this->rigidBody->setMass(this->mass);
        this->rigidBody->setDamping(this->damping, this->damping);
        this->rigidBody->setAwake(true);
        this->rigidBody->setCanSleep(false);
        this->rigidBody->sleepEpsilon = 0.3;

        Athena::Vector3 halfSize(scale.coordinates.x * 0.5, scale.coordinates.y * 0.5 ,scale.coordinates.z * 0.5);
        /*it.data[0] = 1.0 / 12.0 * this->mass * (halfSize.coordinates.y * halfSize.coordinates.y + halfSize.coordinates.z * halfSize.coordinates.z);
        it.data[4] = 1.0 / 12.0 * this->mass * (halfSize.coordinates.x * halfSize.coordinates.x + halfSize.coordinates.z * halfSize.coordinates.z);
        it.data[8] = 1.0 / 12.0 * this->mass * (halfSize.coordinates.x * halfSize.coordinates.x + halfSize.coordinates.y * halfSize.coordinates.y);*/
        this->rigidBody->setInertiaTensor(it);
        this->cBox = new CollisionBox(halfSize);
        cBox->body = this->rigidBody;

        //setupBox();

    }

    void RigidBox::update()
    {
        createBBox();
        //drawBox();
    }

    void RigidBox::startRuntime()
    {
        this->physicSimulation->instance->bodyList.push_back(this->rigidBody);
        this->physicSimulation->instance->collisionGenerator->boxes.push_back(this->cBox);
    }

    void RigidBox::updateRuntime()
    {
        this->sceneObject->transform->position = this->rigidBody->position;
        this->sceneObject->transform->rotation = this->rigidBody->orientation;
        std::cout << "rotation: "; this->rigidBody->orientation.asVector4().print();
    }

    void RigidBox::setOrderOfExecution(const short& newOrderOfExecution)
    {

    }

    int RigidBox::getUniqueID()
    {
        return 7652;
    }

    std::string RigidBox::toString()
    {
        return "RigidBox";
    }

    void RigidBox::createBBox()
    {
        auto localScale = this->sceneObject->transform->localScale;

        GLfloat min_x, max_x, min_y, max_y, min_z, max_z;
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

        this->boundingBox.size = Athena::Vector3((max_x - min_x) * localScale.coordinates.x,
                                                 (max_y - min_y) * localScale.coordinates.y,
                                                 (max_z - min_z) * localScale.coordinates.z);
    }

    void RigidBox::setupBox()
    {
        //this->bboxShader = new Odysseus::Shader(".\\Shader\\bboxShader.vert", ".\\Shader\\bboxShader.frag");
        //bboxShader->use();
        // Cube 1x1x1, centered on origin
        GLfloat vertices[] = {
            -0.5, -0.5, -0.5, 1.0,
            0.5, -0.5, -0.5, 1.0,
            0.5,  0.5, -0.5, 1.0,
            -0.5,  0.5, -0.5, 1.0,
            -0.5, -0.5,  0.5, 1.0,
            0.5, -0.5,  0.5, 1.0,
            0.5,  0.5,  0.5, 1.0,
            -0.5,  0.5,  0.5, 1.0,
        };
        GLuint vbo_vertices;
        glGenBuffers(1, &vbo_vertices);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_vertices);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        GLushort elements[] = {
            0, 1, 2, 3,
            4, 5, 6, 7,
            0, 4, 1, 5, 2, 6, 3, 7
        };
        GLuint ibo_elements;
        glGenBuffers(1, &ibo_elements);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo_elements);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(elements), elements, GL_STATIC_DRAW);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

        //bboxShader->setVec3("position", this->transform->position);
        //bboxShader->setVec3("scale", this->transform->localScale);
        //bboxShader->setVec4("rotation", this->transform->rotation.asVector4());

        glBindBuffer(GL_ARRAY_BUFFER, vbo_vertices);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(
            0,                  // attribute
            4,                  // number of elements per vertex, here (x,y,z,w)
            GL_FLOAT,           // the type of each element
            GL_FALSE,           // take our values as-is
            0,                  // no extra data between each position
            0                   // offset of first element
        );

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo_elements);
        glDrawElements(GL_LINE_LOOP, 4, GL_UNSIGNED_SHORT, 0);
        glDrawElements(GL_LINE_LOOP, 4, GL_UNSIGNED_SHORT, (GLvoid*)(4*sizeof(GLushort)));
        glDrawElements(GL_LINES, 8, GL_UNSIGNED_SHORT, (GLvoid*)(8*sizeof(GLushort)));
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

        glDisableVertexAttribArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        glDeleteBuffers(1, &vbo_vertices);
        glDeleteBuffers(1, &ibo_elements);
    }

    void RigidBox::drawBox()
    {
        bboxShader->use();
        bboxShader->setVec3("position", this->transform->position);
        bboxShader->setVec3("scale", this->transform->localScale);
        bboxShader->setVec4("rotation", this->transform->rotation.asVector4());
    }

    void RigidBox::setPhysicsSimulation(RigidPhysicsEngine* physSim)
    {
        this->physicSimulation = physSim;
    }

    void RigidBox::serialize(YAML::Emitter& out)
    {
        out << YAML::Key << this->toString();
        out << YAML::BeginMap;
            out << YAML::Key << "Size";
            out << YAML::BeginMap;
                out << YAML::Key << "X" << YAML::Value << this->boundingBox.size.coordinates.x;
                out << YAML::Key << "Y" << YAML::Value << this->boundingBox.size.coordinates.y;
                out << YAML::Key << "Z" << YAML::Value << this->boundingBox.size.coordinates.z;
            out << YAML::EndMap;
        out << YAML::EndMap;
    }

    System::Component* RigidBox::deserialize(YAML::Node& node)
    {
        auto component = node[this->toString()];
        this->boundingBox.size = Athena::Vector3();
        this->boundingBox.size.coordinates.x = component["Size"]["X"].as<Athena::Scalar>();
        this->boundingBox.size.coordinates.y = component["Size"]["Y"].as<Athena::Scalar>();
        this->boundingBox.size.coordinates.z = component["Size"]["Z"].as<Athena::Scalar>();

        return this;
    }

    SERIALIZE_CLASS
    {
        System::SerializableClass::registerClass<RigidBox>("RigidBox");
    }
}