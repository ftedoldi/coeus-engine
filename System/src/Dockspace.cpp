#include "../Dockspace.hpp"

#include <Editor.hpp>

#include <SceneManager.hpp>

#include <Folder.hpp>
#include <GUI.hpp>

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
        gameSceneWindow = new EditorLayer::GameSceneWindow();

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
        gameSceneWindow->draw();
    }
}