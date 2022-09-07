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

    // TODO: Draggable Scene Items
    void HierarchyWindow::draw()
    {
        ImGui::Begin("Hierarchy");

        ImVec2 pos = ImGui::GetCursorPos();
        ImGui::Dummy(ImGui::GetContentRegionAvail());
        if (ImGui::BeginDragDropTarget())
        {
            const ImGuiPayload* payload = ImGui::AcceptDragDropPayload(IMGUI_PAYLOAD_TYPE_COLOR_3F, ImGuiDragDropFlags_AcceptBeforeDelivery | ImGuiDragDropFlags_AcceptNoDrawDefaultRect);
                if (payload)
                    if (payload->IsPreview())
                        hoveredDraggingTransform = nullptr;
                    if (payload->IsDelivery() && hoveredDraggingTransform == nullptr)
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

                if (ImGui::IsItemHovered() && ImGui::IsItemClicked(ImGuiMouseButton_Left))
                    selectedItem = Odysseus::SceneManager::activeScene->objectsInScene[i]->transform;

                if (isOpen)
                {
                    if (ImGui::BeginDragDropSource())
                    {
                        ImGui::SetDragDropPayload(IMGUI_PAYLOAD_TYPE_COLOR_3F, nullptr, 0);
                        ImGui::EndDragDropSource();
                    }

                    if (ImGui::BeginDragDropTarget())
                    {
                        const ImGuiPayload* payload = ImGui::AcceptDragDropPayload(IMGUI_PAYLOAD_TYPE_COLOR_3F, ImGuiDragDropFlags_AcceptBeforeDelivery | ImGuiDragDropFlags_AcceptNoDrawDefaultRect);
                        if (payload)
                            if (payload->IsPreview())
                                hoveredDraggingTransform = Odysseus::SceneManager::activeScene->objectsInScene[i]->transform;
                            if (payload->IsDelivery() && hoveredDraggingTransform != nullptr)
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
                                hoveredDraggingTransform->children.push_back(selectedItem);
                            }
                        ImGui::EndDragDropTarget();
                    }

                    System::Utils::GUI::displayChildrenOfTransform(
                                                            Odysseus::SceneManager::activeScene->objectsInScene[i]->transform, 
                                                            Odysseus::SceneManager::activeScene->sceneEditor->selectedTransform,
                                                            this
                                                        );
                }
                else
                {
                    if (ImGui::BeginDragDropSource())
                    {
                        ImGui::SetDragDropPayload(IMGUI_PAYLOAD_TYPE_COLOR_3F, nullptr, 0);
                        ImGui::EndDragDropSource();
                    }
        
                    if (ImGui::BeginDragDropTarget())
                    {
                        const ImGuiPayload* payload = ImGui::AcceptDragDropPayload(IMGUI_PAYLOAD_TYPE_COLOR_3F, ImGuiDragDropFlags_AcceptBeforeDelivery | ImGuiDragDropFlags_AcceptNoDrawDefaultRect);
                        if (payload)
                            if (payload->IsPreview())
                                hoveredDraggingTransform = Odysseus::SceneManager::activeScene->objectsInScene[i]->transform;
                            if (payload->IsDelivery() && hoveredDraggingTransform != nullptr)
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
                                hoveredDraggingTransform->children.push_back(selectedItem);
                            }
                        ImGui::EndDragDropTarget();
                    }
                }
            }
            else
            {
                ImGui::TreeNodeEx(
                    std::to_string(Odysseus::SceneManager::activeScene->objectsInScene[i]->ID).c_str(),
                    ImGuiTreeNodeFlags_CollapsingHeader | ImGuiTreeNodeFlags_Bullet,
                    Odysseus::SceneManager::activeScene->objectsInScene[i]->transform->name.c_str());

                if (ImGui::IsItemHovered() && ImGui::IsItemClicked(ImGuiMouseButton_Left))
                {
                    Odysseus::SceneManager::activeScene->sceneEditor->selectedTransform = Odysseus::SceneManager::activeScene->objectsInScene[i]->transform;
                    selectedItem = Odysseus::SceneManager::activeScene->objectsInScene[i]->transform;
                    System::Utils::GUI::loadInspectorParameters(Odysseus::SceneManager::activeScene->sceneEditor->selectedTransform);
                }

                if (ImGui::BeginDragDropSource())
                {
                    ImGui::SetDragDropPayload(IMGUI_PAYLOAD_TYPE_COLOR_3F, nullptr, 0);
                    ImGui::EndDragDropSource();
                }

                if (ImGui::BeginDragDropTarget())
                {
                    const ImGuiPayload* payload = ImGui::AcceptDragDropPayload(IMGUI_PAYLOAD_TYPE_COLOR_3F, ImGuiDragDropFlags_AcceptBeforeDelivery | ImGuiDragDropFlags_AcceptNoDrawDefaultRect);
                    if (payload)
                        if (payload->IsPreview())
                            hoveredDraggingTransform = Odysseus::SceneManager::activeScene->objectsInScene[i]->transform;
                        if (payload->IsDelivery() && hoveredDraggingTransform != nullptr)
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
                            hoveredDraggingTransform->children.push_back(selectedItem);
                        }
                    ImGui::EndDragDropTarget();
                }
            }
        }
        ImGui::End();
    }

} // namespace EditorLayer
