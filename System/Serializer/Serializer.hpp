#ifndef __SERIALIZER_H__
#define __SERIALIZER_H__

#include <SceneObject.hpp>

#include <coeus.hpp>
#include <PointLight.hpp>
#include <SpotLight.hpp>
#include <DirectionalLight.hpp>
#include <AreaLight.hpp>

#include <ModelBase.hpp>

#include <yaml-cpp/yaml.h>

#include <fstream>
#include <string>
#include <unordered_map>

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

                static void serialzieModel(YAML::Emitter& out, Odysseus::ModelBase* model);

                static void serializePointLight(YAML::Emitter& out, Odysseus::PointLight* light);
                static void serializeSpotLight(YAML::Emitter& out, Odysseus::SpotLight* light);
                static void serializeDirectionalLight(YAML::Emitter& out, Odysseus::DirectionalLight* light);
                static void serializeAreaLight(YAML::Emitter& out, Odysseus::AreaLight* light);

                static Odysseus::PointLight* deserializePointLight(YAML::Node& node);
                static Odysseus::SpotLight* deserializeSpotLight(YAML::Node& node);
                static Odysseus::DirectionalLight* deserializeDirectionalLight(YAML::Node& node);
                static Odysseus::AreaLight* deserializeAreaLight(YAML::Node& node);

                static Athena::Vector2 deserializeVector2(YAML::Node& node);
                static Athena::Vector3 deserializeVector3(YAML::Node& node);
                static Athena::Vector4 deserializeVector4(YAML::Node& node);
                static Athena::Quaternion deserializeQuaternion(YAML::Node& node);

                static void deserializeModel(YAML::Node& node);

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