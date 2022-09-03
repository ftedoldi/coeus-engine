#include "../HierarchyWindow.hpp"

#include <imgui.h>
#include <imgui_internal.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <SceneManager.hpp>

namespace EditorLayer
{
    
    HierarchyWindow::HierarchyWindow()
    {
        Odysseus::Transform *selectedItem = nullptr;
    }
    
    void HierarchyWindow::drawPopupMenu()
    {
        if (ImGui::IsMouseClicked(ImGuiMouseButton_Right))
            ImGui::OpenPopup("PopupMenu");

        if (ImGui::BeginPopup("PopupMenu"))
        {
            if (ImGui::Selectable("Add New SceneObject"))
                Odysseus::SceneObject *o = new Odysseus::SceneObject("New Scene Object");

            if (ImGui::Selectable("Delete SceneObject") && selectedItem != nullptr)
            {
                Odysseus::SceneManager::activeScene->deleteSceneObject(selectedItem->sceneObject);
                Odysseus::SceneManager::activeScene->sceneEditor->selectedTransform = nullptr;
            }

            ImGui::EndPopup();
        }
    }

    // TODO: Draggable Scene Items
    void HierarchyWindow::draw()
    {
        ImGui::Begin("Hierarchy");

        this->drawPopupMenu();

        for (int i = 0; i < Odysseus::SceneManager::activeScene->objectsInScene.size(); i++)
        {
            if (
                Odysseus::SceneManager::activeScene->objectsInScene[i]->transform->parent != nullptr ||
                !Odysseus::SceneManager::activeScene->objectsInScene[i]->showInEditor)
                continue;

            if (Odysseus::Transform::CountNestedChildren(Odysseus::SceneManager::activeScene->objectsInScene[i]->transform))
            {
                auto isOpen = ImGui::TreeNodeEx(
                    std::to_string(Odysseus::SceneManager::activeScene->objectsInScene[i]->ID).c_str(),
                    ImGuiTreeNodeFlags_CollapsingHeader,
                    Odysseus::SceneManager::activeScene->objectsInScene[i]->transform->name.c_str());

                if (ImGui::IsItemHovered() && ImGui::IsItemClicked(ImGuiMouseButton_Left))
                {
                    Odysseus::SceneManager::activeScene->sceneEditor->selectedTransform = Odysseus::SceneManager::activeScene->objectsInScene[i]->transform;
                    System::Utils::GUI::loadInspectorParameters(Odysseus::SceneManager::activeScene->sceneEditor->selectedTransform);
                }

                if (isOpen)
                    System::Utils::GUI::displayChildrenOfTransform(
                                                            Odysseus::SceneManager::activeScene->objectsInScene[i]->transform, 
                                                            Odysseus::SceneManager::activeScene->sceneEditor->selectedTransform
                                                        );
            }
            else
            {
                ImGui::TreeNodeEx(
                    std::to_string(Odysseus::SceneManager::activeScene->objectsInScene[i]->ID).c_str(),
                    ImGuiTreeNodeFlags_CollapsingHeader | ImGuiTreeNodeFlags_Bullet,
                    Odysseus::SceneManager::activeScene->objectsInScene[i]->transform->name.c_str());
                if (ImGui::IsItemHovered() && ImGui::IsItemClicked())
                {
                    Odysseus::SceneManager::activeScene->sceneEditor->selectedTransform = Odysseus::SceneManager::activeScene->objectsInScene[i]->transform;
                    System::Utils::GUI::loadInspectorParameters(Odysseus::SceneManager::activeScene->sceneEditor->selectedTransform);
                }
            }

            if (ImGui::IsItemHovered() && ImGui::IsItemClicked(ImGuiMouseButton_Right))
                selectedItem = Odysseus::SceneManager::activeScene->objectsInScene[i]->transform;
        }
        ImGui::End();
    }

} // namespace EditorLayer
