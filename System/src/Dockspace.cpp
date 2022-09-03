#include "../Dockspace.hpp"

#include <Editor.hpp>

#include <EditorCamera.hpp>
#include <Transform.hpp>
#include <Model.hpp>

#include <HierarchyWindow.hpp>

#include <SerializableClass.hpp>

#include <Serializer/Serializer.hpp>

#include <Utils/WindowUtils.hpp>
#include <Utils/GUI.hpp>

#include <regex>

// TODO: Refactor all of this
// TODO: Implement file cancellation within the editor
namespace System
{
    Dockspace::Dockspace()
    {
        gizmoOperation = ImGuizmo::OPERATION::TRANSLATE;

        statusBar = new EditorLayer::StatusBar();
        mainMenuBar = new EditorLayer::MainMenuBar(this->statusBar);
        toolBar = new EditorLayer::ToolBar(this->statusBar, gizmoOperation);

        hierarchyWindow = new EditorLayer::HierarchyWindow();
        consoleWindow = new EditorLayer::ConsoleWindow();
        contentBrowserWindow = new EditorLayer::ContentBrowserWindow();
        inspectorWindow = new EditorLayer::InspectorWindow();
        projectSettingsWindow = new EditorLayer::ProjectSettingsWindow();
        inspectorSceneWindow = new EditorLayer::InspectorSceneWindow(this->statusBar, gizmoOperation);

        Folder::assetDirectory = Folder::getFolderPath("Assets");
        Folder::currentDirectory = Folder::assetDirectory;
    }

    void Dockspace::draw()
    {
        if (Odysseus::SceneManager::activeScene->isRuntimeScene)
            Utils::GUI::setEngineRuntimeStyle();
        else
            Utils::GUI::setEngineEditorStyle();

        Utils::GUI::createDockableArea();

        mainMenuBar->draw();
        toolBar->draw();
        statusBar->draw();

        ImGui::End();

        hierarchyWindow->draw();
        consoleWindow->draw();
        contentBrowserWindow->draw();
        inspectorWindow->draw();
        inspectorSceneWindow->draw();
        projectSettingsWindow->draw();
        createGameWindow();
    }

    // TODO: setup framebuffer for game scene
    void Dockspace::createGameWindow()
    {
        ImGui::Begin("Game");
        ImGui::BeginChild("Game Render");
        ImDrawList *dList = ImGui::GetWindowDrawList();
        dList->AddCallback((ImDrawCallback)&Window::sceneFrameBuffer->framebufferShaderCallback, Window::sceneFrameBuffer);
        ImGuizmo::SetRect(
            ImGui::GetWindowPos().x, ImGui::GetWindowPos().y,
            Window::sceneFrameBuffer->frameBufferSize.width,
            Window::sceneFrameBuffer->frameBufferSize.height);
        ImVec2 size = ImGui::GetWindowSize();

        // Window::frameBufferSize.width = size.x;
        // Window::frameBufferSize.height = size.y;

        // Window::sceneFrameBuffer->blit();

#pragma warning(push)
#pragma warning(disable : 4312)
        ImGui::Image((ImTextureID)Window::sceneFrameBuffer->texturesID[0], size, ImVec2(0, 1), ImVec2(1, 0));
#pragma warning(pop)

        dList->AddCallback(ImDrawCallback_ResetRenderState, nullptr);
        ImGui::EndChild();
        ImGui::End();
    }
}