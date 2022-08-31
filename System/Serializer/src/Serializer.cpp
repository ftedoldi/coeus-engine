#include "../Serializer.hpp"

#include <SceneObject.hpp>

#include <SceneManager.hpp>

#include <ModelBase.hpp>
#include <Model.hpp>
#include <Shader.hpp>
#include <Scene.hpp>

#include <Transform.hpp>

#include <EditorCameraMovement.hpp>

#include <Mesh.hpp>

namespace YAML
{
    template<>
    struct convert<Athena::Vector2>
    {
        static Node encode(const Athena::Vector2& rhs)
        {
            Node node;

            node.push_back(rhs.coordinates.x);
            node.push_back(rhs.coordinates.y);

            return node;
        }

        static bool decode(const Node& node, Athena::Vector2& rhs)
        {
            rhs.coordinates.x = node["X"].as<float>();
            rhs.coordinates.y = node["Y"].as<float>();

            return true;
        }
    };
    
    template<>
    struct convert<Athena::Vector3>
    {
        static Node encode(const Athena::Vector3& rhs)
        {
            Node node;

            node.push_back(rhs.coordinates.x);
            node.push_back(rhs.coordinates.y);
            node.push_back(rhs.coordinates.z);

            return node;
        }

        static bool decode(const Node& node, Athena::Vector3& rhs)
        {
            rhs.coordinates.x = node["X"].as<float>();
            rhs.coordinates.y = node["Y"].as<float>();
            rhs.coordinates.z = node["Z"].as<float>();

            return true;
        }
    };

    template<>
    struct convert<Athena::Vector4>
    {
        static Node encode(const Athena::Vector4& rhs)
        {
            Node node;

            node.push_back(rhs.coordinates.x);
            node.push_back(rhs.coordinates.y);
            node.push_back(rhs.coordinates.z);
            node.push_back(rhs.coordinates.w);

            return node;
        }

        static bool decode(const Node& node, Athena::Vector4& rhs)
        {
            rhs.coordinates.x = node["X"].as<float>();
            rhs.coordinates.y = node["Y"].as<float>();
            rhs.coordinates.z = node["Z"].as<float>();
            rhs.coordinates.w = node["W"].as<float>();

            return true;
        }
    };

    template<>
    struct convert<Athena::Quaternion>
    {
        static Node encode(const Athena::Quaternion& rhs)
        {
            Node node;

            node.push_back(rhs.immaginary.coordinates.x);
            node.push_back(rhs.immaginary.coordinates.y);
            node.push_back(rhs.immaginary.coordinates.z);
            node.push_back(rhs.real);

            return node;
        }

        static bool decode(const Node& node, Athena::Quaternion& rhs)
        {
            rhs.immaginary.coordinates.x = node["X"].as<float>();
            rhs.immaginary.coordinates.y = node["Y"].as<float>();
            rhs.immaginary.coordinates.z = node["Z"].as<float>();
            rhs.real = node["W"].as<float>();

            return true;
        }
    };

    template<>
    struct convert<std::vector<Odysseus::Vertex>>
    {
        // TODO: Implement me
        static Node encode(const std::vector<Odysseus::Vertex>& rhs)
        {
            Node node;

            return node;
        }

        static bool decode(const Node& node, std::vector<Odysseus::Vertex>& rhs)
        {
            for (int i = 0; i < node["Vertices"].size(); i++)
            {
                Odysseus::Vertex v;

                v.Position = node["Vertices"][i].as<Athena::Vector3>();
                v.Normal = node["Normals"][i].as<Athena::Vector3>();
                v.Tangent = node["Tangents"][i].as<Athena::Vector3>();
                v.TexCoords = node["Texture Coordinates"][i].as<Athena::Vector2>();

                rhs.push_back(v);
            }

            return true;
        }
    };
}

namespace System::Serialize
{

    Serializer::Serializer()
    {

    }

    void Serializer::serialize(const std::string& filepath)
    {
        YAML::Emitter out;

        out << YAML::BeginMap; // Begin map
            out << YAML::Key << "Scene" << YAML::Value << Odysseus::SceneManager::activeScene->name;
            out << YAML::Key << "Scene Path" << YAML::Value << Odysseus::SceneManager::activeScene->path;
            out << YAML::Key << "Scene Objects" << YAML::Value;
            out << YAML::BeginSeq; // Begin sequence of values

                for (auto sceneObject : Odysseus::SceneManager::activeScene->objectsInScene)
                {
                    this->serializeSceneObject(out, *sceneObject);
                }

            out << YAML::EndSeq;
        out << YAML::EndMap;


        std::ofstream fout(filepath);
        fout << out.c_str();
        fout.close();
    }

    void Serializer::serializeSceneObject(YAML::Emitter& out, const Odysseus::SceneObject& objectTosSerialize)
    {
        bool isEditorCamera = false;
        for (auto component : objectTosSerialize._container->components)
        {
            if (component->toString() == "EditorCamera")
                isEditorCamera = true;
        }

        if (isEditorCamera)
            return;

        out << YAML::BeginMap;
            out << YAML::Key << "Scene Object ID" << YAML::Value << objectTosSerialize.ID; // Entity ID goes here

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
                    out << YAML::Key << "Children Scene Objects" << YAML::Value;
                    serializeTransformChildren(out, objectTosSerialize.transform->children);
                }
            out << YAML::EndMap;

            if (objectTosSerialize._container->components.size() > 0)
            {
                out << YAML::Key << "Components" << YAML::Value;
                out << YAML::BeginSeq;
                    // TODO: Add method in order to convert custom component and serialize them
                    for (auto component : objectTosSerialize._container->components)
                    {
                        out << YAML::BeginMap;
                            out << YAML::Key << "Component" << YAML::Value << component->toString();
                            
                            if (component->toString() == "ModelBase")
                                serialzieModel(out, dynamic_cast<Odysseus::ModelBase*>(component));
                            else
                                component->serialize(out);
                        out << YAML::EndMap;
                    }
                out << YAML::EndSeq;
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
            out << YAML::Key << "Z" << YAML::Value << vector.coordinates.z;
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
        out << YAML::BeginSeq;
            for (auto child : children)
                out << child->sceneObject->ID;
        out << YAML::EndSeq;
    }

    void Serializer::serializeRuntime(const std::string& filepath)
    {

    }

    void Serializer::serialzieModel(YAML::Emitter& out, Odysseus::ModelBase* model)
    {
        out << YAML::Key << "Model";
        out << YAML::BeginMap;
            out << YAML::Key << "Path" << YAML::Value << model->modelPath;
            out << YAML::Key << "PBR" << YAML::Value << model->isPBR;
            out << YAML::Key << "Vertex Shader" << YAML::Value << model->vertexShaderPath;
            out << YAML::Key << "Fragment Shader" << YAML::Value << model->fragmentShaderPath;
        out << YAML::EndMap;
    }

    void Serializer::deserializeModel(YAML::Node& node)
    {

    }

    bool Serializer::deserialize(const std::string& filepath)
    {
        std::ifstream stream(filepath);
        std::stringstream strStream;
        strStream << stream.rdbuf();

        YAML::Node data = YAML::Load(strStream.str());
        if (!data["Scene"])
            return false;

        std::string sceneName = data["Scene"].as<std::string>();
        std::string scenePath = data["Scene Path"].as<std::string>();
        std::cout << "Deserializing scene: " << sceneName << std::endl;

        try
        {
                Odysseus::Scene* deserializedScene = new Odysseus::Scene(scenePath, sceneName);
                Odysseus::SceneManager::addScene(deserializedScene);
                Odysseus::SceneManager::activeScene = deserializedScene;
                std::cout << "Scene Name: " << Odysseus::SceneManager::activeScene->name << std::endl;
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << '\n';
        }
        
        // TODO: Fix bug of swapping scenes

        auto sceneObjects = data["Scene Objects"];
        if (!sceneObjects)
        {
            std::cout << "No sceneObjects found in the scene" << std::endl;
            return true;
        }

        // TODO: Use that to store objects added in scene for then set the transform parent tree
        std::unordered_map<std::uint64_t, Odysseus::SceneObject*> objectsInScene;

        for (auto obj : sceneObjects)
        {
            auto ID = obj["Scene Object ID"].as<std::uint64_t>();
            auto transformComponent = obj["Transform"];
            auto name = transformComponent["Name"].as<std::string>();
            auto position = deserializeVector3(transformComponent["Position"]);
            auto rotation = deserializeQuaternion(transformComponent["Rotation"]);
            auto eulerAnglesRotation = deserializeVector3(transformComponent["Euler Angles Rotation"]);
            auto scale = deserializeVector3(transformComponent["Scale"]);
            auto parent = transformComponent["Parent Scene Object"];

            Odysseus::Transform* serializedTransofrm = new Odysseus::Transform(position, rotation, scale);
            serializedTransofrm->name = name;
            serializedTransofrm->eulerRotation = eulerAnglesRotation;
            Odysseus::SceneObject* serializedObject = new Odysseus::SceneObject(*serializedTransofrm, ID);

            objectsInScene[serializedObject->ID] = serializedObject;
            
            // TODO: Deserialize children only if they are not children of model
            auto components = obj["Components"];
            if (components)
            {
                bool isModel = false;
                for (auto component : components)
                {
                    auto model = component["Model"];
                    if (model)
                        isModel = true;
                }
                
                for (auto component : components)
                {
                    auto name = component["Component"].as<std::string>();

                    if (name == "ModelBase")
                        continue;

                    std::cout << name << std::endl;

                    rttr::type t = rttr::type::get_by_name(name);
                    rttr::variant v = t.create();
                    System::Component* c = v.convert<System::Component*>();

                    try
                    {
                        auto componentToAdd = c->deserialize(component);
                        if (componentToAdd != nullptr)
                            serializedObject->addCopyOfExistingComponent<System::Component>(componentToAdd);

                    }
                    catch(const std::exception& e)
                    {
                        std::cerr << e.what() << '\n';
                    }
                    
                }
            }
        }

        for (auto obj : sceneObjects)
        {
            auto ID = obj["Scene Object ID"].as<std::uint64_t>();
            auto transformComponent = obj["Transform"];

            if (objectsInScene[ID]->getComponent<Odysseus::ModelBase>() == nullptr)
            {
                auto parent = transformComponent["Parent Scene Object"];
                if (parent)
                {
                    objectsInScene[ID]->transform->parent = objectsInScene[parent.as<std::uint64_t>()]->transform;
                    objectsInScene[ID]->transform->parent->children.push_back(objectsInScene[ID]->transform);
                }
            }

        }

        Odysseus::SceneManager::initializeActiveScene();
        return true;
    }

    Athena::Vector2 Serializer::deserializeVector2(YAML::Node& node)
    {
        auto x = node["X"].as<Athena::Scalar>();
        auto y = node["Y"].as<Athena::Scalar>();

        return Athena::Vector2(x, y);
    }

    Athena::Vector3 Serializer::deserializeVector3(YAML::Node& node)
    {
        auto x = node["X"].as<Athena::Scalar>();
        auto y = node["Y"].as<Athena::Scalar>();
        auto z = node["Z"].as<Athena::Scalar>();

        return Athena::Vector3(x, y, z);
    }

    Athena::Vector4 Serializer::deserializeVector4(YAML::Node& node)
    {
        auto x = node["X"].as<Athena::Scalar>();
        auto y = node["Y"].as<Athena::Scalar>();
        auto z = node["Z"].as<Athena::Scalar>();
        auto w = node["W"].as<Athena::Scalar>();

        return Athena::Vector4(x, y, z, w);
    }

    Athena::Quaternion Serializer::deserializeQuaternion(YAML::Node& node)
    {
        auto x = node["X"].as<Athena::Scalar>();
        auto y = node["Y"].as<Athena::Scalar>();
        auto z = node["Z"].as<Athena::Scalar>();
        auto w = node["W"].as<Athena::Scalar>();

        return Athena::Quaternion(x, y, z, w);
    }

    bool Serializer::deserializeRuntime(const std::string& filepath)
    {
        return false;
    }

}