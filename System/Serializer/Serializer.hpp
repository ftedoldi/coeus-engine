#ifndef __SERIALIZER_H__
#define __SERIALIZER_H__

#include <coeus.hpp>

#include <yaml-cpp/yaml.h>

#include <fstream>
#include <string>
#include <unordered_map>

namespace Odysseus
{
    class SceneObject;
    class PointLight;
    class SpotLight;
    class DirectionalLight;
    class AreaLight;
    class ModelBase;
    class Transform;
}

namespace System
{
    namespace Serialize
    {
        class Serializer
        {
            private:
                static void serializeSceneObject(YAML::Emitter& out, const Odysseus::SceneObject& objectTosSerialize);

                static void serializeTransformChildren(YAML::Emitter& out, const std::vector<Odysseus::Transform*> children);

                static void serializeVector2(YAML::Emitter& out, const Athena::Vector2& vector);
                static void serializeVector3(YAML::Emitter& out, const Athena::Vector3& vector);
                static void serializeVector4(YAML::Emitter& out, const Athena::Vector4& vector);
                static void serializeQuaternion(YAML::Emitter& out, const Athena::Quaternion& quaternion);

                static void serializeModel(YAML::Emitter& out, Odysseus::ModelBase* model);

                static Athena::Vector2 deserializeVector2(YAML::Node& node);
                static Athena::Vector3 deserializeVector3(YAML::Node& node);
                static Athena::Vector4 deserializeVector4(YAML::Node& node);
                static Athena::Quaternion deserializeQuaternion(YAML::Node& node);

            public:
                Serializer();

                // Serialize the scene at the specified relative file path in a YAML file
                void serialize(const std::string& filepath);
                // Serialize the scene at the specified relative file path in a binary file
                void serializeRuntime(const std::string& filepath);

                // Deserialize the scene at the specified relative file path in a YAML file
                bool deserialize(const std::string& filepath);
                // Deserialize the scene at the specified relative file path in a binary file
                bool deserializeRuntime(const std::string& filepath);
        };
    } // namespace Serialize   
} // namespace System

#endif // __SERIALIZER_H__