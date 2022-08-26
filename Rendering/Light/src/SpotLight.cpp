#include "../SpotLight.hpp"

#include <Folder.hpp>

namespace Odysseus
{
    SpotLight::SpotLight()
    {
        _ambient = Athena::Vector3();
        _diffuse = Athena::Vector3(0.5f, 0.5f, 0.5f);
        _specular = Athena::Vector3();

        _direction = Athena::Vector3(0.5f, 0.5f, 0.5f).normalized();
        _spotExponent = 0.1f;
        _cutOff = 0.1f;
        
        shader = new Odysseus::Shader(".\\Shader\\phongShader.vert", ".\\Shader\\phongShader.frag");

        this->_editorTextureID = Odysseus::Texture2D::loadTextureFromFile(
                                                                                    (System::Folder::getFolderPath("Icons").string() + "/spotLight.png").c_str(), 
                                                                                    true
                                                                                ).ID;
        this->_hasEditorTexture = true;
    }

    Athena::Vector3 SpotLight::getPosition() const
    {
        return this->transform->position;
    }

    Athena::Vector3 SpotLight::getDirection() const
    {
        return this->_direction;
    }

    float SpotLight::getCutOff() const
    {
        return this->_cutOff;
    }

    void SpotLight::setPosition(Athena::Vector3& position)
    {
        this->transform->position = position;
    }

    void SpotLight::setDirection(Athena::Vector3& direction)
    {
        this->_direction = direction;
    }

    void SpotLight::setCutOff(float cutOff)
    {
        this->_cutOff = cutOff;
    }

    void SpotLight::setSpotExponent(float spotExp)
    {
        this->_spotExponent = spotExp;
    }

    void SpotLight::start()
    {

    }
    void SpotLight::update()
    {
        this->setLightShader(this->shader);
    }

    void SpotLight::setOrderOfExecution(const short& newOrderOfExecution)
    {

    }

    int SpotLight::getUniqueID()
    {
        return this->_uniqueID;
    }

    std::string SpotLight::toString()
    {
        return "SpotLight";
    }

    void SpotLight::showComponentFieldsInEditor()
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
        ImGui::InputFloat3("Direction Factor", dir);
        _direction = Athena::Vector3(dir[0], dir[1], dir[2]);
        ImGui::InputFloat("Spot Exponent", &_spotExponent);
        ImGui::InputFloat("CutOff Exponent", &_cutOff);
    }

    void SpotLight::serialize(YAML::Emitter& out)
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
            out << YAML::Key << "Spot Exponent" << YAML::Value << this->_spotExponent;
            out << YAML::Key << "CutOff" << YAML::Value << this->_cutOff;
            out << YAML::Key << "Vertex Shader Path" << YAML::Value << this->shader->vertexShaderPath;
            out << YAML::Key << "Fragment Shader Path" << YAML::Value << this->shader->fragmentShaderPath;
        out << YAML::EndMap;
    }

    System::Component* SpotLight::deserialize(YAML::Node& node)
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
        this->_spotExponent = component["Spot Exponent"].as<float>();
        this->_cutOff = component["CutOff"].as<float>();

        auto vShaderPath= component["Vertex Shader Path"].as<std::string>();
        auto fShaderPath= component["Fragment Shader Path"].as<std::string>();

        this->shader = new Odysseus::Shader(vShaderPath.c_str(), fShaderPath.c_str());

        return this;
    }

    void SpotLight::setLightShader(Odysseus::Shader* shader) const
    {
        //https://math.hws.edu/graphicsbook/c7/s2.html
        //here there is an interective graphic representation
        //of how cutOff angle work with spotExponent
        shader->use();
        auto worldPosition = Transform::GetWorldTransform(this->transform, this->transform);
        
        shader->setVec3("spotLight.diffuse", this->_diffuse);
        shader->setVec3("spotLight.specular", this->_specular);
        shader->setVec3("spotLight.ambient", this->_ambient);
        shader->setVec3("spotLight.position", worldPosition->position);
        shader->setVec3("spotLight.direction", this->_direction);
        shader->setFloat("spotLight.spotExponent", this->_spotExponent);
        //We calculate the cosine value here because its needed in the fragment shader and also because calculating it in the shader would be expensive
        shader->setFloat("spotLight.cutOff", std::cos(Athena::Math::degreeToRandiansAngle(_cutOff)));
    }

    SERIALIZE_CLASS
    {
        System::SerializableClass::registerClass<SpotLight>("SpotLight");
    }
}