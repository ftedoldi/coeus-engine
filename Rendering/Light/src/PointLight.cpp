#include "../PointLight.hpp"

#include <AreaLight.hpp>

#include <SceneManager.hpp>
#include <EditorCamera.hpp>

#include <Texture2D.hpp>
#include <Folder.hpp>

#include <LightInfo.hpp>

namespace Odysseus
{
    PointLight::PointLight()
    {
        _ambient = Athena::Vector3();
        _diffuse = Athena::Vector3(0.5f, 0.5f, 0.5f);
        _specular = Athena::Vector3();

        _constant = 0.1f;
        _linear = 0.1f;
        _quadratic = 0.1f;

        this->ID = System::UUID();

        this->_editorTextureID = Odysseus::Texture2D::loadTextureFromFile(
                                                                                    (System::Folder::getFolderPath("Icons").string() + "/pointLight.png").c_str(), 
                                                                                    true
                                                                                ).ID;
        this->_hasEditorTexture = true;
    }

    Athena::Vector3 PointLight::getPosition() const
    {
        return this->transform->position;
    }

    float PointLight::getConstant() const
    {
        return this->_constant;
    }

    float PointLight::getLinear() const
    {
        return this->_linear;
    }

    float PointLight::getQuadratic() const
    {
        return this->_quadratic;
    }

    void PointLight::setPosition(Athena::Vector3& position)
    {
        this->transform->position = position;
    }

    void PointLight::setConstant(float constant)
    {
        this->_constant = constant;
    }

    void PointLight::setLinear(float linear)
    {
        this->_linear = linear;
    }

    void PointLight::setQuadratic(float quadratic)
    {
        this->_quadratic = quadratic;
    }

    void PointLight::start()
    {
        if (LightInfo::existingPointLights.count(this) == 0)
        {
            LightInfo::pointLights.push_back(this);
            LightInfo::existingPointLights.insert(this);
        }
    }

    void PointLight::update()
    {

    }

    void PointLight::setOrderOfExecution(const short& newOrderOfExecution)
    {

    }

    int PointLight::getUniqueID()
    {
        return this->_uniqueID;
    }

    std::string PointLight::toString()
    {
        return "PointLight";
    }

    void PointLight::showComponentFieldsInEditor()
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
        ImGui::InputFloat("Constant Factor", &_constant);
        ImGui::InputFloat("Linear Factor", &_linear);
        ImGui::InputFloat("Quadratic Factor", &_quadratic);
    }

    void PointLight::serialize(YAML::Emitter& out)
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
            out << YAML::Key << "Constant" << YAML::Value << this->_constant;
            out << YAML::Key << "Linear" << YAML::Value << this->_linear;
            out << YAML::Key << "Quadratic" << YAML::Value << this->_quadratic;
        out << YAML::EndMap;
    }

    System::Component* PointLight::deserialize(YAML::Node& node)
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
        this->_constant = component["Constant"].as<float>();
        this->_linear = component["Linear"].as<float>();
        this->_quadratic = component["Quadratic"].as<float>();

        return this;
    }

    void PointLight::setLightShader(Odysseus::Shader* shader) const
    {
        auto tmp = Odysseus::SceneManager::activeScene->sceneEditor->editorCamera->getViewTransform(this->transform);
        auto worldPosition = Transform::GetWorldTransform(this->transform, this->transform);
        shader->use();

        shader->setVec3("pointLight.diffuse", this->_diffuse);
        shader->setVec3("pointLight.specular", this->_specular);
        shader->setVec3("pointLight.ambient", this->_ambient);
        shader->setVec3("pointLight.position", worldPosition->position);
        shader->setFloat("pointLight.constant", this->_constant);
        shader->setFloat("pointLight.linear", this->_linear);
        shader->setFloat("pointLight.quadratic", this->_quadratic);
    }

    void PointLight::setLightShader(Odysseus::Shader* shader, int index) const
    {
        auto tmp = Odysseus::SceneManager::activeScene->sceneEditor->editorCamera->getViewTransform(this->transform);
        auto worldPosition = Transform::GetWorldTransform(this->transform, this->transform);
        shader->use();

        shader->setVec3("pointLights[" + std::to_string(index) + "].diffuse", this->_diffuse);
        shader->setVec3("pointLights[" + std::to_string(index) + "].specular", this->_specular);
        shader->setVec3("pointLights[" + std::to_string(index) + "].ambient", this->_ambient);
        shader->setVec3("pointLights[" + std::to_string(index) + "].position", worldPosition->position);
        shader->setFloat("pointLights[" + std::to_string(index) + "].constant", this->_constant);
        shader->setFloat("pointLights[" + std::to_string(index) + "].linear", this->_linear);
        shader->setFloat("pointLights[" + std::to_string(index) + "].quadratic", this->_quadratic);
    }

    PointLight::~PointLight()
    {   
        int indexToErease = -1;
        for (int i = 0; i < LightInfo::pointLights.size(); i++)
        {
            if (LightInfo::pointLights[i] == this)
            {
                indexToErease = i;
                break;
            }
        }

        if (indexToErease > -1)
        {
            LightInfo::pointLights.erase(LightInfo::pointLights.begin() + indexToErease);
            LightInfo::existingPointLights.erase(this);
        }
    }

    SERIALIZE_CLASS
    {
        System::SerializableClass::registerClass<PointLight>("PointLight");
    }
}