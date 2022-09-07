#include "../DirectionalLight.hpp"

#include <EditorCamera.hpp>

#include <Texture2D.hpp>
#include <Folder.hpp>

#include <LightInfo.hpp>

namespace Odysseus
{

    DirectionalLight::DirectionalLight()
    {
        _ambient = Athena::Vector3();
        _diffuse = Athena::Vector3(0.5f, 0.5f, 0.5f);
        _specular = Athena::Vector3();

        _direction = Athena::Vector3(0.5f, 0.5f, 0.5f).normalized();
        
        this->ID = System::UUID();

        this->_editorTextureID = Odysseus::Texture2D::loadTextureFromFile(
                                                                            (System::Folder::getFolderPath("Icons").string() + "/directionalLight.png").c_str(), 
                                                                            true
                                                                        ).ID;
        this->_hasEditorTexture = true;
    }
    Athena::Vector3 DirectionalLight::getDirection() const
    {
        return this->_direction;
    }

    void DirectionalLight::setDirection(Athena::Vector3& dir)
    {
        this->_direction = dir;
    }

    void DirectionalLight::start()
    {
        if (LightInfo::existingDirectionalLights.count(this) == 0)
        {
            LightInfo::directionalLights.push_back(this);
            LightInfo::existingDirectionalLights.insert(this);
        }
    }
    void DirectionalLight::update()
    {

    }

    void DirectionalLight::setOrderOfExecution(const short& newOrderOfExecution)
    {

    }

    int DirectionalLight::getUniqueID()
    {
        return this->_uniqueID;
    }

    std::string DirectionalLight::toString()
    {
        return "DirectionalLight";
    }

    void DirectionalLight::showComponentFieldsInEditor()
    {
        // TODO: Clamp values between 0 and 1
        float diff[] = { _diffuse.coordinates.x, _diffuse.coordinates.y, _diffuse.coordinates.z };
        ImGui::InputFloat3("Diffuse Factor", diff);
        _diffuse = Athena::Vector3(diff[0], diff[1], diff[2]);
        float amb[] = { _ambient.coordinates.x, _ambient.coordinates.y, _ambient.coordinates.z };
        ImGui::InputFloat3("Ambient Factor", amb);
        _ambient = Athena::Vector3(amb[0], amb[1], amb[2]);
        float spec[] = { _specular.coordinates.x, _specular.coordinates.y, _specular.coordinates.z };
        ImGui::InputFloat3("Specular Factor", spec);
        _specular = Athena::Vector3(spec[0], spec[1], spec[2]);
        float dir[] = { _direction.coordinates.x, _direction.coordinates.y, _direction.coordinates.z };
        ImGui::InputFloat3("Direction", dir);
        _direction = Athena::Vector3(dir[0], dir[1], dir[2]);
    }

    void DirectionalLight::serialize(YAML::Emitter& out)
    {
        out << YAML::Key << this->toString();
        out << YAML::BeginMap;
            out << YAML::Key << "Diffuse";
            out << YAML::BeginMap;
                out << YAML::Key << "Red" << YAML::Value << _diffuse.coordinates.x;
                out << YAML::Key << "Green" << YAML::Value << _diffuse.coordinates.y;
                out << YAML::Key << "Blue" << YAML::Value << _diffuse.coordinates.z;
            out << YAML::EndMap;
            out << YAML::Key << "Ambient";
            out << YAML::BeginMap;
                out << YAML::Key << "Red" << YAML::Value << _ambient.coordinates.x;
                out << YAML::Key << "Green" << YAML::Value << _ambient.coordinates.y;
                out << YAML::Key << "Blue" << YAML::Value << _ambient.coordinates.z;
            out << YAML::EndMap;
            out << YAML::Key << "Specular";
            out << YAML::BeginMap;
                out << YAML::Key << "Red" << YAML::Value << _specular.coordinates.x;
                out << YAML::Key << "Green" << YAML::Value << _specular.coordinates.y;
                out << YAML::Key << "Blue" << YAML::Value << _specular.coordinates.z;
            out << YAML::EndMap;
            out << YAML::Key << "Direction";
            out << YAML::BeginMap;
                out << YAML::Key << "X" << YAML::Value << _direction.coordinates.x;
                out << YAML::Key << "Y" << YAML::Value << _direction.coordinates.y;
                out << YAML::Key << "Z" << YAML::Value << _direction.coordinates.z;
            out << YAML::EndMap;
        out << YAML::EndMap;
    }

    System::Component* DirectionalLight::deserialize(YAML::Node& node)
    {
        auto component = node[this->toString()];

        this->_diffuse = Athena::Vector3();
        this->_diffuse.coordinates.x = component["Diffuse"]["Red"].as<float>();
        this->_diffuse.coordinates.y = component["Diffuse"]["Green"].as<float>();
        this->_diffuse.coordinates.z = component["Diffuse"]["Blue"].as<float>();
        this->_ambient = Athena::Vector3();
        this->_ambient.coordinates.x = component["Ambient"]["Red"].as<float>();
        this->_ambient.coordinates.y = component["Ambient"]["Green"].as<float>();
        this->_ambient.coordinates.z = component["Ambient"]["Blue"].as<float>();
        this->_specular = Athena::Vector3();
        this->_specular.coordinates.x = component["Specular"]["Red"].as<float>();
        this->_specular.coordinates.y = component["Specular"]["Green"].as<float>();
        this->_specular.coordinates.z = component["Specular"]["Blue"].as<float>();
        this->_direction = Athena::Vector3();
        this->_direction.coordinates.x = component["Direction"]["X"].as<float>();
        this->_direction.coordinates.y = component["Direction"]["Y"].as<float>();
        this->_direction.coordinates.z = component["Direction"]["Z"].as<float>();

        return this;
    }

    void DirectionalLight::setLightShader(Odysseus::Shader* shader) const
    {
        shader->use();
        
        shader->setVec3("dirLight.diffuse", this->_diffuse);
        shader->setVec3("dirLight.specular", this->_specular);
        shader->setVec3("dirLight.ambient", this->_ambient);
        shader->setVec3("dirLight.direction", this->_direction);
    }

    void DirectionalLight::setLightShader(Odysseus::Shader* shader, int index) const
    {
        auto tmp = Odysseus::SceneManager::activeScene->sceneEditor->editorCamera->getViewTransform(this->transform);
        auto worldPosition = Transform::GetWorldTransform(this->transform, this->transform);
        shader->use();

        shader->setVec3("directionalLights[" + std::to_string(index) + "].diffuse", this->_diffuse);
        shader->setVec3("directionalLights[" + std::to_string(index) + "].specular", this->_specular);
        shader->setVec3("directionalLights[" + std::to_string(index) + "].ambient", this->_ambient);
        shader->setVec3("directionalLights[" + std::to_string(index) + "].position", worldPosition->position);
        shader->setVec3("directionalLights[" + std::to_string(index) + "].direction", this->_direction);
    }

    DirectionalLight::~DirectionalLight()
    {
        int indexToErease = -1;
        for (int i = 0; i < LightInfo::directionalLights.size(); i++)
        {
            if (LightInfo::directionalLights[i] == this)
            {
                indexToErease = i;
                break;
            }
        }

        if (indexToErease > -1)
        {
            LightInfo::directionalLights.erase(LightInfo::directionalLights.begin() + indexToErease);
            LightInfo::existingDirectionalLights.erase(this);
        }
    }

    SERIALIZE_CLASS
    {
        System::SerializableClass::registerClass<DirectionalLight>("DirectionalLight");
    }
}
