#include "../InspectorWindow.hpp"

#include <Window.hpp>

#include <Component.hpp>

#include <imgui.h>
#include <imgui_internal.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <Folder.hpp>
#include <GUI.hpp>

#include <Transform.hpp>

#include <rttr/registration>

namespace EditorLayer
{
    
    InspectorWindow::InspectorWindow()
    {
        this->setupIcons();
    }

    void InspectorWindow::setupIcons()
    {
        stbi_set_flip_vertically_on_load(false);

        icons.pointLightTextureID = Odysseus::Texture2D::loadTextureFromFile(
                                               (System::Folder::getFolderPath("Icons").string() + "/pointLight.png").c_str(),
                                               true)
                                               .ID;
        icons.spotLightTextureID = Odysseus::Texture2D::loadTextureFromFile(
                                              (System::Folder::getFolderPath("Icons").string() + "/spotLight.png").c_str(),
                                              true)
                                              .ID;
        icons.directionalLightTextureID = Odysseus::Texture2D::loadTextureFromFile(
                                                     (System::Folder::getFolderPath("Icons").string() + "/directionalLight.png").c_str(),
                                                     true)
                                                     .ID;
        icons.areaLightTextureID = Odysseus::Texture2D::loadTextureFromFile(
                                              (System::Folder::getFolderPath("Icons").string() + "/areaLight.png").c_str(),
                                              true)
                                              .ID;
        icons.removeComponentTextureID = Odysseus::Texture2D::loadTextureFromFile(
                                                    (System::Folder::getFolderPath("Icons").string() + "/remove.png").c_str(),
                                                    true)
                                                    .ID;
                                          
        stbi_set_flip_vertically_on_load(true);
    }

    void InspectorWindow::draw()
    {
        static Odysseus::Transform *lastTransform = nullptr;

        ImGui::Begin("Inspector");
        if (Odysseus::SceneManager::activeScene->sceneEditor->selectedTransform != nullptr)
        {
            static char transformName[256] = {0};
            if (Odysseus::SceneManager::activeScene->sceneEditor->selectedTransform != lastTransform)
                strcpy(transformName, Odysseus::SceneManager::activeScene->sceneEditor->selectedTransform->name.c_str());

            ImGui::InputText("Name", transformName, sizeof(char) * 256, ImGuiInputTextFlags_AutoSelectAll);

            if (ImGui::IsItemActive())
                Odysseus::SceneManager::activeScene->sceneEditor->selectedTransform->name = transformName;

            if (glfwGetKey(System::Window::window, GLFW_KEY_ENTER))
                ImGui::SetWindowFocus(NULL);

            ImGui::Separator();
            ImGui::Text("Transform");
            float pos[] = {
                                Odysseus::SceneManager::activeScene->sceneEditor->selectedTransform->position.coordinates.x, 
                                Odysseus::SceneManager::activeScene->sceneEditor->selectedTransform->position.coordinates.y, 
                                Odysseus::SceneManager::activeScene->sceneEditor->selectedTransform->position.coordinates.z
                            };
            ImGui::InputFloat3("Position", pos);
            Odysseus::SceneManager::activeScene->sceneEditor->selectedTransform->position = Athena::Vector3(pos[0], pos[1], pos[2]);

            Athena::Vector3 rotationAsVector = Odysseus::SceneManager::activeScene->sceneEditor->selectedTransform->eulerRotation;
            float rotation[3] = {rotationAsVector[0], rotationAsVector[1], rotationAsVector[2]};
            ImGui::InputFloat3("Rotation", rotation);
            Odysseus::SceneManager::activeScene->sceneEditor->selectedTransform->eulerRotation = Athena::Vector3(rotation[0], rotation[1], rotation[2]);
            if (ImGui::IsItemEdited())
                Odysseus::SceneManager::activeScene->sceneEditor->selectedTransform->rotation = 
                    Athena::Quaternion::EulerAnglesToQuaternion(
                                                                    Odysseus::SceneManager::activeScene->sceneEditor->selectedTransform->eulerRotation
                                                                );

            float scale[] = {
                                Odysseus::SceneManager::activeScene->sceneEditor->selectedTransform->localScale.coordinates.x, 
                                Odysseus::SceneManager::activeScene->sceneEditor->selectedTransform->localScale.coordinates.y, 
                                Odysseus::SceneManager::activeScene->sceneEditor->selectedTransform->localScale.coordinates.z
                            };
            ImGui::InputFloat3("Scale", scale);
            Odysseus::SceneManager::activeScene->sceneEditor->selectedTransform->localScale = Athena::Vector3(scale[0], scale[1], scale[2]);

            // TODO: Do this for every component
            // TODO: Show components serializable fields with protocol buffers
            ImGui::Separator();
            for (int i = 0; i < System::Utils::GUI::inspectorParameters.size(); i++)
            {
#pragma warning(push)
#pragma warning(disable : 4312)
                if (System::Utils::GUI::inspectorParameters[i]->hasEditorTexture())
                {
                    ImGui::Image((ImTextureID)System::Utils::GUI::inspectorParameters[i]->getEditorTextureID(), {12, 12});
                    ImGui::SameLine();
                }
#pragma warning(pop)

                ImGui::Text(System::Utils::GUI::inspectorParameters[i]->toString().c_str());
                ImGui::SameLine(ImGui::GetContentRegionAvail().x - 12);
                ImGui::PushStyleColor(ImGuiCol_Button, {0, 0, 0, 0});
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, {0, 0, 0, 0});
                ImGui::PushStyleColor(ImGuiCol_ButtonActive, {0, 0, 0, 0});
#pragma warning(push)
#pragma warning(disable : 4312)
                bool shouldDeleteComponent = ImGui::ImageButtonEx(
                    i + 1,
                    (ImTextureID)icons.removeComponentTextureID,
                    {12, 12},
                    {0, 0},
                    {1, 1},
                    {0, 2},
                    {0, 0, 0, 0},
                    {1, 1, 1, 1});
#pragma warning(pop)
                System::Utils::GUI::inspectorParameters[i]->showComponentFieldsInEditor();
                ImGui::PopStyleColor(3);
                ImGui::Separator();

                if (shouldDeleteComponent)
                {
                    std::cout << "Deleting: " << System::Utils::GUI::inspectorParameters[i]->toString() << std::endl;
                    Odysseus::SceneManager::activeScene->sceneEditor->selectedTransform->sceneObject->removeComponentWithIndex(i);
                    System::Utils::GUI::loadInspectorParameters(Odysseus::SceneManager::activeScene->sceneEditor->selectedTransform);
                }
            }

            if (System::Utils::GUI::inspectorParameters.size() > Odysseus::SceneManager::activeScene->sceneEditor->selectedTransform->
                    sceneObject->_container->components.size())
                System::Utils::GUI::loadInspectorParameters(Odysseus::SceneManager::activeScene->sceneEditor->selectedTransform);

            static bool isAddComponentOpen = false;

            if (lastTransform != Odysseus::SceneManager::activeScene->sceneEditor->selectedTransform)
            {
                isAddComponentOpen = false;
                lastTransform = Odysseus::SceneManager::activeScene->sceneEditor->selectedTransform;
            }

            // TODO: Remove component button
            if (ImGui::Button("Add Component", {ImGui::GetContentRegionAvail().x, 0}))
            {
                isAddComponentOpen = true;
                ImGui::OpenPopup("Component Popup");
            }

            if (isAddComponentOpen)
            {
#pragma warning(push)
#pragma warning(disable : 4312)
                if (ImGui::BeginPopup("Component Popup"))
                {
                    rttr::type componentType = rttr::type::get_by_name("Component");

                    for (auto derived : componentType.get_derived_classes())
                    {
                        rttr::type t = rttr::type::get_by_name(derived.get_name());
                        rttr::variant v = t.create();

                        System::Component *newComponent = v.convert<System::Component *>();

                        std::string derivedID("##title" + derived.get_name().to_string());
                        auto componentSelectable = ImGui::Selectable(derivedID.c_str());
                        if (newComponent->hasEditorTexture())
                        {
                            ImGui::SameLine();
                            ImGui::Image((ImTextureID)newComponent->getEditorTextureID(), {12, 12});
                        }
                        ImGui::SameLine();
                        ImGui::Text(derived.get_name().to_string().c_str());

                        if (componentSelectable)
                        {
                            auto tmp = Odysseus::SceneManager::activeScene->sceneEditor->
                                        selectedTransform->sceneObject->addCopyOfExistingComponent<System::Component>(newComponent);
                            tmp->start();
                            System::Utils::GUI::inspectorParameters.push_back(tmp);
                        }
                        else
                        {
                            delete newComponent;
                        }
                    }

                    ImGui::EndPopup();
                }
#pragma warning(pop)
            }
        }
        ImGui::End();
    }

} // namespace EditorLayer
