#include "../Serializer.hpp"

#include <SceneObject.hpp>

#include <SceneManager.hpp>

#include <PointLight.hpp>
#include <SpotLight.hpp>
#include <DirectionalLight.hpp>
#include <AreaLight.hpp>

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
            out << YAML::Key << "Scene Objects" << YAML::Value;
            out << YAML::BeginSeq; // Begin sequence of values

                for (auto sceneObject : Odysseus::SceneManager::activeScene->objectsInScene)
                {
                    if (sceneObject->getComponent<Odysseus::Mesh>() == nullptr)
                        this->serializeSceneObject(out, *sceneObject);
                }

            out << YAML::EndSeq;
        out << YAML::EndMap;


        std::ofstream fout(filepath);
        fout << out.c_str();
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
                            
                            if (component->toString() == "PointLight")
                                serializePointLight(out, dynamic_cast<Odysseus::PointLight*>(component));   
                            else if (component->toString() == "DirectionalLight")
                                serializeDirectionalLight(out, dynamic_cast<Odysseus::DirectionalLight*>(component));
                            else if (component->toString() == "AreaLight")
                                serializeAreaLight(out, dynamic_cast<Odysseus::AreaLight*>(component));   
                            else if (component->toString() == "ModelBase")
                                serialzieModel(out, dynamic_cast<Odysseus::ModelBase*>(component));
                            else
                            {
                                component->serialize(out);
                            }
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
        out << YAML::BeginSeq;
            for (auto child : children)
                out << child->sceneObject->ID;
        out << YAML::EndSeq;
    }

    void Serializer::serializeRuntime(const std::string& filepath)
    {

    }

    void Serializer::serializePointLight(YAML::Emitter& out, Odysseus::PointLight* light)
    {
        out << YAML::Key << "PointLight";
        out << YAML::BeginMap;
            out << YAML::Key << "Constant" << YAML::Value << light->_constant; 
            out << YAML::Key << "Linear" << YAML::Value << light->_linear; 
            out << YAML::Key << "Quadratic" << YAML::Value << light->_quadratic; 
            out << YAML::Key << "Diffuse";
            serializeVector3(out, light->_diffuse);
            out << YAML::Key << "Ambient";
            serializeVector3(out, light->_ambient);
            out << YAML::Key << "Specular";
            serializeVector3(out, light->_specular);
            out << YAML::Key << "Vertex Shader Path" << YAML::Value << light->shader->vertexShaderPath;
            out << YAML::Key << "Fragment Shader Path" << YAML::Value << light->shader->fragmentShaderPath;
        out << YAML::EndMap;
    }

    void Serializer::serializeSpotLight(YAML::Emitter& out, Odysseus::SpotLight* light)
    {
        out << YAML::Key << "SpotLight";
        out << YAML::BeginMap;
            out << YAML::Key << "Cut Off" << YAML::Value << light->_cutOff; 
            out << YAML::Key << "Exponent" << YAML::Value << light->_spotExponent; 
            out << YAML::Key << "Direction";
            serializeVector3(out, light->_direction); 
            out << YAML::Key << "Diffuse";
            serializeVector3(out, light->_diffuse);
            out << YAML::Key << "Ambient";
            serializeVector3(out, light->_ambient);
            out << YAML::Key << "Specular";
            serializeVector3(out, light->_specular);
            out << YAML::Key << "Vertex Shader Path" << YAML::Value << light->shader->vertexShaderPath;
            out << YAML::Key << "Fragment Shader Path" << YAML::Value << light->shader->fragmentShaderPath;
        out << YAML::EndMap;   
    }

    void Serializer::serializeDirectionalLight(YAML::Emitter& out, Odysseus::DirectionalLight* light)
    {
        out << YAML::Key << "DirectionalLight";
        out << YAML::BeginMap;
            out << YAML::Key << "Direction";
            serializeVector3(out, light->_direction);
            out << YAML::Key << "Diffuse";
            serializeVector3(out, light->_diffuse);
            out << YAML::Key << "Ambient";
            serializeVector3(out, light->_ambient);
            out << YAML::Key << "Specular";
            serializeVector3(out, light->_specular);
            out << YAML::Key << "Vertex Shader Path" << YAML::Value << light->shader->vertexShaderPath;
            out << YAML::Key << "Fragment Shader Path" << YAML::Value << light->shader->fragmentShaderPath;
        out << YAML::EndMap;   
    }

    void Serializer::serializeAreaLight(YAML::Emitter& out, Odysseus::AreaLight* light)
    {
        out << YAML::Key << "AreaLight";
        out << YAML::BeginMap;
            out << YAML::Key << "Point Lights" << YAML::Value;
            out << YAML::BeginSeq;
                for (auto l : light->pointLights)
                {
                    out << YAML::Key << "PointLight";
                    out << YAML::BeginMap;
                        out << YAML::Key << "Constant" << YAML::Value << l->_constant; 
                        out << YAML::Key << "Linear" << YAML::Value << l->_linear; 
                        out << YAML::Key << "Quadratic" << YAML::Value << l->_quadratic; 
                        out << YAML::Key << "Diffuse";
                        serializeVector3(out, l->_diffuse);
                        out << YAML::Key << "Ambient";
                        serializeVector3(out, l->_ambient);
                        out << YAML::Key << "Specular";
                        serializeVector3(out, l->_specular);
                        out << YAML::Key << "Vertex Shader Path" << YAML::Value << l->shader->vertexShaderPath;
                        out << YAML::Key << "Fragment Shader Path" << YAML::Value << l->shader->fragmentShaderPath;
                    out << YAML::EndMap;
                }
            out << YAML::EndSeq;
            out << YAML::Key << "Diffuse";
            serializeVector3(out, light->_diffuse);
            out << YAML::Key << "Ambient";
            serializeVector3(out, light->_ambient);
            out << YAML::Key << "Specular";
            serializeVector3(out, light->_specular);
            out << YAML::Key << "Vertex Shader Path" << YAML::Value << light->shader->vertexShaderPath;
            out << YAML::Key << "Fragment Shader Path" << YAML::Value << light->shader->fragmentShaderPath;
        out << YAML::EndMap;
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

    Odysseus::PointLight* Serializer::deserializePointLight(YAML::Node& node)
    {
        auto constant = node["Constant"].as<float>();
        auto linear = node["Linear"].as<float>();
        auto quadratic = node["Quadratic"].as<float>();
        auto diffuseVector = node["Diffuse"];
        auto diffuse = Athena::Vector3(
                                            diffuseVector["X"].as<float>(), 
                                            diffuseVector["Y"].as<float>(), 
                                            diffuseVector["Z"].as<float>()
                                        );
        auto ambientVector = node["Ambient"];
        auto ambient = Athena::Vector3(
                                            ambientVector["X"].as<float>(), 
                                            ambientVector["Y"].as<float>(), 
                                            ambientVector["Z"].as<float>()
                                        );
        auto specularVector = node["Specular"];
        auto specular = Athena::Vector3(
                                            specularVector["X"].as<float>(), 
                                            specularVector["Y"].as<float>(), 
                                            specularVector["Z"].as<float>()
                                        );

        auto vShaderPath = node["Vertex Shader Path"].as<std::string>();
        auto fShaderPath = node["Fragment Shader Path"].as<std::string>();

        Odysseus::PointLight* pLight = new Odysseus::PointLight();
        pLight->_constant = constant;
        pLight->_linear = linear;
        pLight->_quadratic = quadratic;

        pLight->_diffuse = diffuse;
        pLight->_ambient = ambient;
        pLight->_specular = specular;
        pLight->shader = new Odysseus::Shader(vShaderPath.c_str(), fShaderPath.c_str());

        return pLight;
    }

    Odysseus::SpotLight* Serializer::deserializeSpotLight(YAML::Node& node)
    {
        auto exponent = node["Exponent"].as<float>();
        auto cutoff = node["Cut Off"].as<float>();
        auto directionVector = node["Direction"];
        auto direction = Athena::Vector3(
                                            directionVector["X"].as<float>(), 
                                            directionVector["Y"].as<float>(), 
                                            directionVector["Z"].as<float>()
                                        );
        auto diffuseVector = node["Diffuse"];
        auto diffuse = Athena::Vector3(
                                            diffuseVector["X"].as<float>(), 
                                            diffuseVector["Y"].as<float>(), 
                                            diffuseVector["Z"].as<float>()
                                        );
        auto ambientVector = node["Ambient"];
        auto ambient = Athena::Vector3(
                                            ambientVector["X"].as<float>(), 
                                            ambientVector["Y"].as<float>(), 
                                            ambientVector["Z"].as<float>()
                                        );
        auto specularVector = node["Specular"];
        auto specular = Athena::Vector3(
                                            specularVector["X"].as<float>(), 
                                            specularVector["Y"].as<float>(), 
                                            specularVector["Z"].as<float>()
                                        );

        auto vShaderPath = node["Vertex Shader Path"].as<std::string>();
        auto fShaderPath = node["Fragment Shader Path"].as<std::string>();

        Odysseus::SpotLight* sLight = new Odysseus::SpotLight();
        sLight->_spotExponent = exponent;
        sLight->_cutOff = cutoff;
        sLight->_direction = direction;

        sLight->_diffuse = diffuse;
        sLight->_ambient = ambient;
        sLight->_specular = specular;
        sLight->shader = new Odysseus::Shader(vShaderPath.c_str(), fShaderPath.c_str());

        return sLight;
    }

    Odysseus::DirectionalLight* Serializer::deserializeDirectionalLight(YAML::Node& node)
    {
        auto directionVector = node["Direction"];
        auto direction = Athena::Vector3(
                                            directionVector["X"].as<float>(), 
                                            directionVector["Y"].as<float>(), 
                                            directionVector["Z"].as<float>()
                                        );
        auto diffuseVector = node["Diffuse"];
        auto diffuse = Athena::Vector3(
                                            diffuseVector["X"].as<float>(), 
                                            diffuseVector["Y"].as<float>(), 
                                            diffuseVector["Z"].as<float>()
                                        );
        auto ambientVector = node["Ambient"];
        auto ambient = Athena::Vector3(
                                            ambientVector["X"].as<float>(), 
                                            ambientVector["Y"].as<float>(), 
                                            ambientVector["Z"].as<float>()
                                        );
        auto specularVector = node["Specular"];
        auto specular = Athena::Vector3(
                                            specularVector["X"].as<float>(), 
                                            specularVector["Y"].as<float>(), 
                                            specularVector["Z"].as<float>()
                                        );

        auto vShaderPath = node["Vertex Shader Path"].as<std::string>();
        auto fShaderPath = node["Fragment Shader Path"].as<std::string>();

        Odysseus::DirectionalLight* dLight = new Odysseus::DirectionalLight();
        dLight->_direction = direction;

        dLight->_diffuse = diffuse;
        dLight->_ambient = ambient;
        dLight->_specular = specular;
        dLight->shader = new Odysseus::Shader(vShaderPath.c_str(), fShaderPath.c_str());

        return dLight;
    }

    // TODO: Implement me
    Odysseus::AreaLight* Serializer::deserializeAreaLight(YAML::Node& node)
    {
        return nullptr;
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
        std::cout << "Deserializing scene: " << sceneName << std::endl;

        Odysseus::Scene* deserializedScene = new Odysseus::Scene(sceneName);
        Odysseus::SceneManager::addScene(deserializedScene);
        std::cout << Odysseus::SceneManager::setActiveScene(0) << std::endl;
        std::cout << "Scene Name: " << Odysseus::SceneManager::activeScene->name << std::endl;

        Editor* editor = new Editor();
        Odysseus::SceneManager::activeScene->sceneEditor = editor;

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

                if (!isModel)
                {
                    Odysseus::Transform* serializedTransofrm = new Odysseus::Transform(position, rotation, scale);
                    serializedTransofrm->name = name;
                    serializedTransofrm->eulerRotation = eulerAnglesRotation;
                    Odysseus::SceneObject* serializedObject = new Odysseus::SceneObject(*serializedTransofrm, ID);

                    objectsInScene[serializedObject->ID] = serializedObject;

                    for (auto component : components)
                    {
                        auto name = component["Component"].as<std::string>();

                        rttr::type t = rttr::type::get_by_name(name);
                        rttr::variant v = t.create();
                        System::Component* c = v.convert<System::Component*>();

                        try
                        {
                            if (name == "PointLight")
                                serializedObject->addCopyOfExistingComponent<Odysseus::PointLight>(deserializePointLight(component["PointLight"]));
                            else if (name == "DirectionalLight")
                                serializedObject->addCopyOfExistingComponent<Odysseus::DirectionalLight>(deserializeDirectionalLight(component["DirectionalLight"]));
                            else if (name == "AreaLight")
                                serializedObject->addCopyOfExistingComponent<Odysseus::AreaLight>(deserializeAreaLight(component["AreaLight"]));
                            else
                            {
                                auto componentToAdd = c->deserialize(component);
                                if (componentToAdd != nullptr)
                                    serializedObject->addCopyOfExistingComponent<System::Component>(componentToAdd);
                            }

                        }
                        catch(const std::exception& e)
                        {
                            std::cerr << e.what() << '\n';
                        }
                        
                    }
                }
                else
                {
                    auto name = components[0]["Component"].as<std::string>();
                    
                    auto model = components[0]["Model"];
                    if (model)
                    {
                        auto vShaderPath = model["Vertex Shader"].as<std::string>();
                        auto fShaderPath = model["Fragment Shader"].as<std::string>();

                        Odysseus::Shader* shader = new Odysseus::Shader(vShaderPath.c_str(), fShaderPath.c_str());

                        bool isPBR = model["PBR"].as<bool>();

                        std::cout << "Is PBR: " << isPBR << std::endl;

                        std::string modelPath = model["Path"].as<std::string>();

                        Odysseus::Model model(modelPath, shader, isPBR);
                    }
                }
            }
            else if (!parent)
            {
                // TODO: Fix name deserialization issues
                Odysseus::Transform* serializedTransofrm = new Odysseus::Transform(position, rotation, scale);
                serializedTransofrm->name = name;
                serializedTransofrm->eulerRotation = eulerAnglesRotation;
                Odysseus::SceneObject* serializedObject = new Odysseus::SceneObject(*serializedTransofrm, ID);

                objectsInScene[serializedObject->ID] = serializedObject;
            }
        }

        // for (auto obj : sceneObjects)
        // {
        //     auto ID = obj["Scene Object ID"].as<std::uint64_t>();
        //     auto transformComponent = obj["Transform"];

        //     std::cout << objectsInScene[ID]->transform->name;

        //     if (objectsInScene[ID]->getComponent<Odysseus::ModelBase>() == nullptr)
        //     {
        //         auto parent = transformComponent["Parent Scene Object"];
        //         if (parent)
        //             objectsInScene[ID]->transform->parent = objectsInScene[parent.as<std::uint64_t>()]->transform;

        //         auto children = transformComponent["Children Scene Objects"];
        //         if (children)
        //             for (auto c : children)
        //             {
        //                 auto childID = c.as<std::uint64_t>();
        //                 objectsInScene[ID]->transform->children.push_back(objectsInScene[childID]->transform);
        //             }
        //     }

        // }

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