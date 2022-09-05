#include "../AreaLight.hpp"

#include <Texture2D.hpp>
#include <Folder.hpp>

namespace Odysseus
{
    AreaLight::AreaLight()
    {
        _ambient = Athena::Vector3();
        _diffuse = Athena::Vector3(0.5f, 0.5f, 0.5f);
        _specular = Athena::Vector3();

        auto pLight = new PointLight();

        pointLights.push_back(pLight);
        
        shader = new Odysseus::Shader(".\\Shader\\phongShader.vert", ".\\Shader\\phongShader.frag");

        this->_editorTextureID = Odysseus::Texture2D::loadTextureFromFile(
                                                                            (System::Folder::getFolderPath("Icons").string() + "/areaLight.png").c_str(),
                                                                            true
                                                                        ).ID;
        this->_hasEditorTexture = true;
    }
    void AreaLight::start()
    {
        unsigned int numLights = this->pointLights.size();
        shader->setInt("numLights", numLights);
    }
    void AreaLight::update()
    {
        this->setLightShader(this->shader);
    }

    void AreaLight::setOrderOfExecution(const short& newOrderOfExecution)
    {

    }

    int AreaLight::getUniqueID()
    {
        return this->_uniqueID;
    }

    std::string AreaLight::toString()
    {
        return "AreaLight";
    }

    void AreaLight::showComponentFieldsInEditor()
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
        for (int i = 0; i < this->pointLights.size(); i++)
        {
            float lightPos[] = {    
                                    this->pointLights[i]->transform->position.coordinates.x, 
                                    this->pointLights[i]->transform->position.coordinates.y, 
                                    this->pointLights[i]->transform->position.coordinates.z
                                };
            ImGui::InputFloat3(std::string("Point Light" + i).c_str(), lightPos);
            this->pointLights[i]->transform->position = Athena::Vector3(lightPos[0], lightPos[1], lightPos[2]);
        }
    }

    void AreaLight::serialize(YAML::Emitter& out)
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
            out << YAML::Key << "Point Lights" << YAML::Value;
            out << YAML::BeginSeq;
                for (auto pLight : pointLights)
                {
                    pLight->serialize(out);
                }
            out << YAML::EndSeq;
            out << YAML::Key << "Vertex Shader Path" << YAML::Value << this->shader->vertexShaderPath;
            out << YAML::Key << "Fragment Shader Path" << YAML::Value << this->shader->fragmentShaderPath;
        out << YAML::EndMap;
    }

    // TODO: Check this implementation
    // FIXME: Check this implementation
    System::Component* AreaLight::deserialize(YAML::Node& node)
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
        
        for (auto pLight : component["PointLights"])
        {
            PointLight* newPLight = new PointLight();
            newPLight->deserialize(pLight);
            this->addLight(newPLight);
        }

        auto vShaderPath= component["Vertex Shader Path"].as<std::string>();
        auto fShaderPath= component["Fragment Shader Path"].as<std::string>();

        this->shader = new Odysseus::Shader(vShaderPath.c_str(), fShaderPath.c_str());

        return this;
    }

    void AreaLight::addLight(PointLight* pt)
    {
        this->pointLights.push_back(pt);
    }

    void AreaLight::setLightShader(Odysseus::Shader* shader) const
    {
        this->shader->use();
        
        for(unsigned int i = 0; i < this->pointLights.size(); ++i)
        {
            std::cout << "pointlight" << i << " position: ";
            pointLights[i]->getPosition().print();
            std::cout << std::endl;
            shader->setVec3("pointLights[i].position", pointLights[i]->getPosition());
            shader->setVec3("pointLights[i].ambient", pointLights[i]->getAmbient());
            shader->setVec3("pointLights[i].diffuse", pointLights[i]->getDiffuse());
            shader->setVec3("pointLights[i].specular", pointLights[i]->getSpecular());
            shader->setFloat("pointLights[i].constant", pointLights[i]->getConstant());
            shader->setFloat("pointLights[i].linear", pointLights[i]->getLinear());
            shader->setFloat("pointLights[i].quadratic", pointLights[i]->getQuadratic());
        }
    }

    SERIALIZE_CLASS
    {
        System::SerializableClass::registerClass<AreaLight>("AreaLight");
    }
}
