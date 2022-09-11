#include "../ToolBar.hpp"

#include <Window.hpp>

#include <IO/Input.hpp>
#include <Folder.hpp>
#include <UUID.hpp>

namespace EditorLayer
{
    ToolBar::ToolBar(StatusBar*& mainStatusBar, ImGuizmo::OPERATION& currentGizmoOperation) : gizmoOperation(currentGizmoOperation)
    {
        this->_mainStatusBar = mainStatusBar;
        this->initializeIcons();
    }

    void ToolBar::initializeIcons()
    {
        this->icons.translateTextureID = Odysseus::Texture2D::loadTextureFromFile(
                                                                                    (System::Folder::getFolderPath("Icons").string() + "/translate.png").c_str(), 
                                                                                    true
                                                                                ).ID;
        this->icons.rotateTextureID = Odysseus::Texture2D::loadTextureFromFile(
                                                                                    (System::Folder::getFolderPath("Icons").string() + "/rotation.png").c_str(), 
                                                                                    true
                                                                                ).ID;
        this->icons.scaleTextureID = Odysseus::Texture2D::loadTextureFromFile(
                                                                                    (System::Folder::getFolderPath("Icons").string() + "/resize.png").c_str(), 
                                                                                    true
                                                                                ).ID;
        this->icons.playTextureID = Odysseus::Texture2D::loadTextureFromFile(
                                                                                    (System::Folder::getFolderPath("Icons").string() + "/play.png").c_str(), 
                                                                                    true
                                                                                ).ID;
        this->icons.pauseTextureID = Odysseus::Texture2D::loadTextureFromFile(
                                                                                    (System::Folder::getFolderPath("Icons").string() + "/pause.png").c_str(), 
                                                                                    true
                                                                                ).ID;
        this->icons.stopTextureID = Odysseus::Texture2D::loadTextureFromFile(
                                                                                    (System::Folder::getFolderPath("Icons").string() + "/stop.png").c_str(), 
                                                                                    true
                                                                                ).ID;
    }

    void ToolBar::setStatusBar(StatusBar*& mainStatusBar)
    {
        this->_mainStatusBar = mainStatusBar;
    }

    void ToolBar::initializeShortcuts()
    {
        if (glfwGetKey(System::Window::window, GLFW_KEY_T))
            this->translateObject();
        if (glfwGetKey(System::Window::window, GLFW_KEY_R))
            this->rotateObject();
        if (glfwGetKey(System::Window::window, GLFW_KEY_S))
            this->scaleObject();

        if (
                glfwGetKey(System::Window::window, GLFW_KEY_LEFT_CONTROL) && 
                glfwGetKey(System::Window::window, GLFW_KEY_P) && 
                !Odysseus::SceneManager::activeScene->isRuntimeScene
            )
            this->playScene();

        if (
                glfwGetKey(System::Window::window, GLFW_KEY_LEFT_ALT) && 
                glfwGetKey(System::Window::window, GLFW_KEY_P) && 
                Odysseus::SceneManager::activeScene->isRuntimeScene &&
                Odysseus::SceneManager::activeScene->status == Odysseus::SceneState::RUNNING
            )
            this->pauseScene();
        else if (
                    glfwGetKey(System::Window::window, GLFW_KEY_LEFT_ALT) && 
                    glfwGetKey(System::Window::window, GLFW_KEY_P) && 
                    Odysseus::SceneManager::activeScene->isRuntimeScene &&
                    Odysseus::SceneManager::activeScene->status == Odysseus::SceneState::PAUSED
                )
            this->resumeScene();

        if (
                glfwGetKey(System::Window::window, GLFW_KEY_LEFT_ALT) && 
                glfwGetKey(System::Window::window, GLFW_KEY_S) && 
                Odysseus::SceneManager::activeScene->isRuntimeScene &&
                Odysseus::SceneManager::activeScene->status != Odysseus::SceneState::EDITOR
            )
            this->stopScene();
    }

    void ToolBar::translateObject()
    {
        gizmoOperation = ImGuizmo::OPERATION::TRANSLATE;
        _mainStatusBar->addStatus("Translate operation selected", EditorLayer::StatusBarTextColor::GREEN);
    }

    void ToolBar::rotateObject()
    {
        gizmoOperation = ImGuizmo::OPERATION::ROTATE;
        _mainStatusBar->addStatus("Rotate operation selected", EditorLayer::StatusBarTextColor::GREEN);
    }

    void ToolBar::scaleObject()
    {
        gizmoOperation = ImGuizmo::OPERATION::SCALE;
        _mainStatusBar->addStatus("Scale operation selected", EditorLayer::StatusBarTextColor::GREEN);
    }

    void ToolBar::playScene()
    {
        _mainStatusBar->addStatus("Starting simulation...");
        Odysseus::Scene* simulationScene = new Odysseus::Scene(Odysseus::SceneManager::activeScene, Odysseus::SceneState::RUNNING, true);
        Odysseus::SceneManager::activeScene = simulationScene;
        Odysseus::SceneManager::activeScene->initialiseScene();
    }

    void ToolBar::pauseScene()
    {
        _mainStatusBar->addStatus("Pausing simulation...");
        Odysseus::SceneManager::activeScene->status = Odysseus::SceneState::PAUSED;
    }

    void ToolBar::resumeScene()
    {
        _mainStatusBar->addStatus("Resuming simulation...");
        Odysseus::SceneManager::activeScene->status = Odysseus::SceneState::RUNNING;
    }

    void ToolBar::stopScene()
    {
        _mainStatusBar->addStatus("Stopping simulation...");
        auto serializedScene = Odysseus::SceneManager::activeScene;
        Odysseus::SceneManager::activeScene->status = Odysseus::SceneState::STOPPED;
        System::Serialize::Serializer serializer = System::Serialize::Serializer();
        serializer.deserialize(Odysseus::SceneManager::_loadedScenes[Odysseus::SceneManager::_loadedScenes.size() - 1]->path);
        Odysseus::SceneManager::_loadedScenes.pop_back();
        delete serializedScene;
    }

    // TODO: Initialize textures for buttons
    void ToolBar::draw()
    {
        this->initializeShortcuts();
        
        ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.43f, 0.43f, 0.50f, 0));
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, {2, 8});
        
        if (
            ImGui::BeginViewportSideBar(
                                            "Tool Bar", 
                                            ImGui::GetMainViewport(), 
                                            ImGuiDir_Up, 
                                            8, 
                                            ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_MenuBar
                                        )
            )
        {
            if (ImGui::BeginMenuBar()) {
                    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.38f, 0.38f, 0.38f, 0.50f));
                    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.38f, 0.38f, 0.38f, 0.70f));
                    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.67f, 0.67f, 0.67f, 0.39f));

                    static ImVec2 buttonPadding = ImVec2(4, 4);
                    static ImVec2 buttonSize = ImVec2(18, 18);

                    ImVec2 cursor = ImGui::GetCursorPos();
                    ImGui::SetCursorPos({ cursor.x + 0.5f, cursor.y + 2.5f });

                    #pragma warning(push)
                    #pragma warning(disable : 4312)                
                    ImGui::ImageButtonEx(
                                System::UUID(),
                                (ImTextureID)this->icons.translateTextureID, 
                                buttonSize,
                                { 0, 0 },
                                { 1, 1 },
                                buttonPadding,
                                { 0, 0, 0, 0 },
                                { 1, 1, 1, 1 }
                        );
                    #pragma warning(pop)

                    if (ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left))
                        this->translateObject();

                    ImGui::SameLine();
                    cursor = ImGui::GetCursorPos();
                    ImGui::SetCursorPos({ cursor.x, cursor.y });

                    #pragma warning(push)
                    #pragma warning(disable : 4312)                
                    ImGui::ImageButtonEx(
                                System::UUID(),
                                (ImTextureID)this->icons.scaleTextureID, 
                                buttonSize,
                                { 0, 0 },
                                { 1, 1 },
                                buttonPadding,
                                { 0, 0, 0, 0 },
                                { 1, 1, 1, 1 }
                        );
                    #pragma warning(pop)

                    if (ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left))
                        this->scaleObject();
                    
                    ImGui::SameLine();
                    cursor = ImGui::GetCursorPos();
                    ImGui::SetCursorPos({ cursor.x, cursor.y });

                    #pragma warning(push)
                    #pragma warning(disable : 4312)                
                    ImGui::ImageButtonEx(
                                System::UUID(),
                                (ImTextureID)this->icons.rotateTextureID, 
                                buttonSize,
                                { 0, 0 },
                                { 1, 1 },
                                buttonPadding,
                                { 0, 0, 0, 0 },
                                { 1, 1, 1, 1 }
                        );
                    #pragma warning(pop)

                    if (ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left))
                        this->rotateObject();

                    ImGui::SameLine();
                    cursor = ImGui::GetCursorPos();
                    ImGui::SetCursorPos({ ImGui::GetContentRegionAvail().x / 2, cursor.y });

                    #pragma warning(push)
                    #pragma warning(disable : 4312)                
                    ImGui::ImageButtonEx(
                                System::UUID(),
                                (ImTextureID)this->icons.playTextureID, 
                                buttonSize,
                                { 0, 0 },
                                { 1, 1 },
                                buttonPadding,
                                { 0, 0, 0, 0 },
                                { 1, 1, 1, 1 }
                        );
                    #pragma warning(pop)

                    if (
                            (ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) &&
                            !Odysseus::SceneManager::activeScene->isRuntimeScene
                        )
                        this->playScene();
                    
                    ImGui::SameLine();

                    #pragma warning(push)
                    #pragma warning(disable : 4312)                
                    ImGui::ImageButtonEx(
                                System::UUID(),
                                (ImTextureID)this->icons.pauseTextureID, 
                                buttonSize,
                                { 0, 0 },
                                { 1, 1 },
                                buttonPadding,
                                { 0, 0, 0, 0 },
                                { 1, 1, 1, 1 }
                        );
                    #pragma warning(pop)

                    if (    
                            (ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) && 
                            Odysseus::SceneManager::activeScene->isRuntimeScene &&
                            Odysseus::SceneManager::activeScene->status == Odysseus::SceneState::RUNNING
                        )
                        this->pauseScene();
                    else if (
                                (ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) && 
                                Odysseus::SceneManager::activeScene->isRuntimeScene &&
                                Odysseus::SceneManager::activeScene->status == Odysseus::SceneState::PAUSED
                            )
                        this->resumeScene();
                    
                    ImGui::SameLine();

                    #pragma warning(push)
                    #pragma warning(disable : 4312)                
                    ImGui::ImageButtonEx(
                                System::UUID(),
                                (ImTextureID)this->icons.stopTextureID, 
                                buttonSize,
                                { 0, 0 },
                                { 1, 1 },
                                buttonPadding,
                                { 0, 0, 0, 0 },
                                { 1, 1, 1, 1 }
                        );
                    #pragma warning(pop)

                    if (
                            (ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) && 
                            Odysseus::SceneManager::activeScene->isRuntimeScene &&
                            Odysseus::SceneManager::activeScene->status != Odysseus::SceneState::EDITOR
                        )
                        this->stopScene();

                    ImGui::PopStyleColor(3);
                ImGui::EndMenuBar();
            }
            ImGui::End();
        }

        ImGui::PopStyleColor();
        ImGui::PopStyleVar();
    }
} // namespace EditorLayer
