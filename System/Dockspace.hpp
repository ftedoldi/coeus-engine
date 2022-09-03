#ifndef __DOCKSPACE_H__
#define __DOCKSPACE_H__

#include <Component.hpp>

#include "Window.hpp"

#include <imgui.h>
#include <imgui_internal.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <ImGuizmo.h>

#include <IO/Input.hpp>
#include <Folder.hpp>

#include <StatusBar.hpp>
#include <MainMenuBar.hpp>
#include <ToolBar.hpp>
#include <ConsoleWindow.hpp>
#include <ContentBrowserWindow.hpp>
#include <InspectorWindow.hpp>
#include <ProjectSettingsWindow.hpp>
#include <InspectorSceneWindow.hpp>

#include "Debug.hpp"

#include <SceneManager.hpp>
#include <Shader.hpp>
#include <Texture2D.hpp>
#include <Math.hpp>
#include <Matrix4.hpp>

#include <iostream>
#include <string>
#include <cmath>
#include <filesystem>
#include <vector>
#include <algorithm>
#include <random>

#include <stb/stb_image.h>

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

            ImGuizmo::OPERATION gizmoOperation;

            // TODO: Refactor this
            //----------------------Window Creation------------------------------//
            void createGameWindow();

        public:
            Dockspace();
            
            void draw();
    };
}

#endif // __DOCKSPACE_H__