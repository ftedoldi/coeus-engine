#include "../DummyComponent.hpp"

#include <imgui.h>
#include <imgui_internal.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <Texture2D.hpp>
#include <Folder.hpp>

DummyComponent::DummyComponent()
{
    // _showComponentInEditor = true;
    var = 10;

    this->_editorTextureID = Odysseus::Texture2D::loadTextureFromFile(
                                                                        (System::Folder::getFolderPath("Icons").string() + "/document.png").c_str(),
                                                                        true
                                                                    ).ID;
    this->_hasEditorTexture = true;
}

void DummyComponent::start()
{

}

void DummyComponent::update()
{
    
}

void DummyComponent::setOrderOfExecution(const short& newOrderOfExecution)
{
    _orderOfExecution = newOrderOfExecution;
}

int DummyComponent::getUniqueID()
{
    return _uniqueID;
}

void DummyComponent::showComponentFieldsInEditor()
{
    ImGui::InputFloat(NAMEOF(asd), &asd);
    ImGui::InputInt(NAMEOF(var), &var);
}

void DummyComponent::serialize(YAML::Emitter& out)
{
    out << YAML::Key << this->toString();
    out << YAML::BeginMap;
        out << YAML::Key << NAMEOF(asd) << YAML::Value << this->asd;
        out << YAML::Key << NAMEOF(var) << YAML::Value << this->var;
    out << YAML::EndMap; 
}

System::Component* DummyComponent::deserialize(YAML::Node& node)
{
    auto component = node[this->toString()];
    this->asd = component["asd"].as<int>();
    this->var = component["var"].as<int>();

    return this;
}

std::string DummyComponent::toString()
{
    return "DummyComponent";
}

DummyComponent::~DummyComponent()
{

}

RTTR_REGISTRATION
{
    System::SerializableClass::registerClass<DummyComponent>("DummyComponent");
}
