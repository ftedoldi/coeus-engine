#include "../HierarchyWindow.hpp"

#include <imgui.h>
#include <imgui_internal.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <SceneManager.hpp>
#include <GUI.hpp>

namespace EditorLayer
{
    
    HierarchyWindow::HierarchyWindow()
    {
        selectedItem = nullptr;
        hoveredDraggingTransform = nullptr;
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

    void HierarchyWindow::beginDragAndDropWindowEmptyArea()
    {
        ImVec2 pos = ImGui::GetCursorPos();
        ImGui::Dummy(ImGui::GetContentRegionAvail());
        if (ImGui::BeginDragDropTarget())
        {
            const ImGuiPayload* payload = ImGui::AcceptDragDropPayload(IMGUI_PAYLOAD_TYPE_COLOR_3F, ImGuiDragDropFlags_AcceptBeforeDelivery | ImGuiDragDropFlags_AcceptNoDrawDefaultRect);
                if (payload)
                    if (payload->IsDelivery() && hoveredDraggingTransform == nullptr && !ImGui::IsAnyItemHovered())
                    {
                        if (selectedItem->parent)
                        {
                            auto parent = selectedItem->parent;
                            int indexToPop = -1;
                            for (int i = 0; i < parent->children.size(); i++)
                                if (parent->children[i] == selectedItem)
                                    indexToPop = i;
                            parent->children.erase(parent->children.begin() + indexToPop);
                        }

                        selectedItem->parent = hoveredDraggingTransform;
                    }

            ImGui::EndDragDropTarget();
        }
        ImGui::SetCursorPos(pos);
        
        if (ImGui::IsMouseDragging(ImGuiMouseButton_Left) && !ImGui::IsAnyItemHovered())
            hoveredDraggingTransform = nullptr;
    }

    // TODO: Draggable Scene Items
    void HierarchyWindow::draw()
    {
        ImGui::Begin("Hierarchy");

        this->beginDragAndDropWindowEmptyArea();
        
        this->drawPopupMenu();

        for (int i = 0; i < Odysseus::SceneManager::activeScene->objectsInScene.size(); i++)
        {
            if (
                Odysseus::SceneManager::activeScene->objectsInScene[i]->transform->parent != nullptr ||
                !Odysseus::SceneManager::activeScene->objectsInScene[i]->showInEditor)
                continue;

            // TODO: Refactor drag and drop code that is a mess
            if (Odysseus::Transform::CountNestedChildren(Odysseus::SceneManager::activeScene->objectsInScene[i]->transform))
            {
                auto isOpen = ImGui::TreeNodeEx(
                    std::to_string(Odysseus::SceneManager::activeScene->objectsInScene[i]->ID).c_str(),
                    ImGuiTreeNodeFlags_CollapsingHeader,
                    Odysseus::SceneManager::activeScene->objectsInScene[i]->transform->name.c_str());

                System::Utils::GUI::selectTransform(this, Odysseus::SceneManager::activeScene->objectsInScene[i]->transform);
                System::Utils::GUI::beginDragAndDroppableTransform(this, Odysseus::SceneManager::activeScene->objectsInScene[i]->transform);

                if (isOpen)
                {
                    System::Utils::GUI::displayChildrenOfTransform(
                                                            Odysseus::SceneManager::activeScene->objectsInScene[i]->transform,
                                                            this
                                                        );
                }
            }
            else
            {
                ImGui::TreeNodeEx(
                    std::to_string(Odysseus::SceneManager::activeScene->objectsInScene[i]->ID).c_str(),
                    ImGuiTreeNodeFlags_CollapsingHeader | ImGuiTreeNodeFlags_Bullet,
                    Odysseus::SceneManager::activeScene->objectsInScene[i]->transform->name.c_str());

                System::Utils::GUI::selectTransform(this, Odysseus::SceneManager::activeScene->objectsInScene[i]->transform);
                System::Utils::GUI::beginDragAndDroppableTransform(this, Odysseus::SceneManager::activeScene->objectsInScene[i]->transform);
            }
        }
        ImGui::End();
    }
} // namespace EditorLayer
