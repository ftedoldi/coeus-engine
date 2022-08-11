#ifndef __SERIALIZER_H__
#define __SERIALIZER_H__

#include <SceneObject.hpp>

#include <coeus.hpp>

#include <yaml-cpp/yaml.h>

#include <fstream>
#include <string>

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