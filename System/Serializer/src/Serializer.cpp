#include "../Serializer.hpp"

#include <SceneGraph.hpp>

namespace System::Serialize
{

    Serializer::Serializer()
    {

    }

    void Serializer::serialize(const std::string& filepath)
    {
        YAML::Emitter out;

        out << YAML::BeginMap; // Begin map
            out << YAML::Key << "Scene" << YAML::Value << Odysseus::SceneGraph::name;
            out << YAML::Key << "Scene Objects" << YAML::Value;
            out << YAML::BeginSeq; // Begin sequence of values

                for (auto sceneObject : Odysseus::SceneGraph::objectsInScene)
                {
                    this->serializeSceneObject(out, *sceneObject);
                }

            out << YAML::EndSeq;
        out << YAML::EndMap;


        std::ofstream fout(filepath);
        fout << out.c_str();
    }

    void Serializer::serializeSceneObject(YAML::Emitter& out, const Odysseus::SceneObject& objectTosSerialize)
    {
        out << YAML::BeginMap;
            out << YAML::Key << "Scene Object" << YAML::Value << objectTosSerialize.ID; // Entity ID goes here

            out << YAML::Key << "Transform" << YAML::Value;
            out << YAML::BeginMap;
                out << YAML::Key << "Name" << YAML::Value << objectTosSerialize.transform->name;
                out << YAML::Key << "Position";
                serializeVector3(out, objectTosSerialize.transform->position);
                out << YAML::Key << "Rotation";
                serializeQuaternion(out, objectTosSerialize.transform->rotation);
                out << YAML::Key << "Euler Angles Rotation";
                serializeVector3(out, objectTosSerialize.transform->eulerRotation);
                out << YAML::Key << "Scale";
                serializeVector3(out, objectTosSerialize.transform->localScale);
                if (objectTosSerialize.transform->parent != nullptr)
                    out << YAML::Key << "Parent Scene Object" << YAML::Value << objectTosSerialize.transform->parent->sceneObject->ID;
                if (objectTosSerialize.transform->children.size() > 0)
                {
                    out << YAML::Key << "Children Scene Objects";
                    serializeTransformChildren(out, objectTosSerialize.transform->children);
                }
            out << YAML::EndMap;
            
            // TODO: Add hasmap that holds all the possible Components of the Engine
            // TODO: Add serialization of the components parameters
            for (auto component : objectTosSerialize._container->components)
            {
                out << YAML::Key << "Component";
                out << YAML::BeginMap;
                    out << YAML::Key << "Name" << YAML::Value << component->toString();
                out << YAML::EndMap;   
            }
        out << YAML::EndMap;
    }

    void Serializer::serializeVector2(YAML::Emitter& out, const Athena::Vector2& vector)
    {
        out << YAML::BeginMap;
            out << YAML::Key << "X" << YAML::Value << vector.coordinates.x;
            out << YAML::Key << "Y" << YAML::Value << vector.coordinates.y;
        out << YAML::EndMap;
    }

    void Serializer::serializeVector3(YAML::Emitter& out, const Athena::Vector3& vector)
    {
        out << YAML::BeginMap;
            out << YAML::Key << "X" << YAML::Value << vector.coordinates.x;
            out << YAML::Key << "Y" << YAML::Value << vector.coordinates.y;
            out << YAML::Key << "Z" << YAML::Value << vector.coordinates.x;
        out << YAML::EndMap;
    }

    void Serializer::serializeVector4(YAML::Emitter& out, const Athena::Vector4& vector)
    {
        out << YAML::BeginMap;
            out << YAML::Key << "X" << YAML::Value << vector.coordinates.x;
            out << YAML::Key << "Y" << YAML::Value << vector.coordinates.y;
            out << YAML::Key << "Z" << YAML::Value << vector.coordinates.z;
            out << YAML::Key << "W" << YAML::Value << vector.coordinates.w;
        out << YAML::EndMap;
    }

    void Serializer::serializeQuaternion(YAML::Emitter& out, const Athena::Quaternion& quaternion)
    {
        out << YAML::BeginMap;
            out << YAML::Key << "X" << YAML::Value << quaternion.immaginary.coordinates.x;
            out << YAML::Key << "Y" << YAML::Value << quaternion.immaginary.coordinates.y;
            out << YAML::Key << "Z" << YAML::Value << quaternion.immaginary.coordinates.z;
            out << YAML::Key << "W" << YAML::Value << quaternion.real;
        out << YAML::EndMap;
    }

    void Serializer::serializeTransformChildren(YAML::Emitter& out, const std::vector<Odysseus::Transform*> children)
    {
        out << YAML::BeginMap;
            for (auto child : children)
                out << YAML::Key << "Scene Object" << YAML::Value << child->sceneObject->ID;
        out << YAML::EndMap;
    }

    void Serializer::serializeRuntime(const std::string& filepath)
    {

    }

    bool Serializer::deserialize(const std::string& filepath)
    {
        return false;
    }

    bool Serializer::deserializeRuntime(const std::string& filepath)
    {
        return false;
    }

}