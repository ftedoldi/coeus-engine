#ifndef __DOCKSPACE_H__
#define __DOCKSPACE_H__

#include <imgui.h>
#include <imgui_internal.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <ImGuizmo.h>

#include <StatusBar.hpp>
#include <MainMenuBar.hpp>
#include <ToolBar.hpp>
#include <ConsoleWindow.hpp>
#include <ContentBrowserWindow.hpp>
#include <InspectorWindow.hpp>
#include <HierarchyWindow.hpp>
#include <ProjectSettingsWindow.hpp>
#include <InspectorSceneWindow.hpp>
#include <GameSceneWindow.hpp>

namespace Odysseus {
    class Transform;
}

namespace EditorLayer
{
    class Editor;
    class HierarchyWindow;
}

namespace System {
    class Window;
    class Component;
    class Console;
    class StatusBar;

    class Dockspace {
        friend class EditorLayer::Editor;
        
        private:
            EditorLayer::StatusBar* statusBar;
            EditorLayer::MainMenuBar* mainMenuBar;
            EditorLayer::ToolBar* toolBar;

            EditorLayer::HierarchyWindow* hierarchyWindow;
            EditorLayer::ConsoleWindow* consoleWindow;
            EditorLayer::ContentBrowserWindow* contentBrowserWindow;
            EditorLayer::InspectorWindow* inspectorWindow;
            EditorLayer::ProjectSettingsWindow* projectSettingsWindow;
            EditorLayer::InspectorSceneWindow* inspectorSceneWindow;
            EditorLayer::GameSceneWindow* gameSceneWindow;

            ImGuizmo::OPERATION gizmoOperation;

        public:
            Dockspace();
            
            void draw();
    };
}

#endif // __DOCKSPACE_H__