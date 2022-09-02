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
#include <Console.hpp>
#include <MainMenuBar.hpp>

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

namespace System {
    class Window;
    class Component;
    class Console;
    class StatusBar;
    
    struct ButtonImages {
        int translateTextureID;
        int rotateTextureID;
        int scaleTextureID;
        int leftArrowTextureID;
        int reloadTextureID;
        int playTextureID;
        int pauseTextureID;
        int stopTextureID;
        int folderTextureID;
        int documentTextureID;
        int pointLightTextureID;
        int spotLightTextureID;
        int directionalLightTextureID;
        int areaLightTextureID;
        int removeComponentTextureID;
        int sceneTextureID;
        int modelTextureID;
    };

    class Dockspace {
        private:
            EditorLayer::Console* console;
            EditorLayer::StatusBar* statusBar;
            EditorLayer::MainMenuBar* mainMenuBar;

            ButtonImages buttonImages;

            ImGuizmo::OPERATION gizmoOperation;

            //----------------------Menu Creation--------------------------------//
            void createToolMenuBar();

            //----------------------Window Creation------------------------------//
            void createHierarchyWindow();
            void createInspectorWindow();
            void createConsoleWindow();
            void createContentBrowser();
            void createSceneWindow();
            void createGameWindow();
            void createProjectSettingsWindow();

            //----------------------Mouse Picking--------------------------------//
            void handleMousePicking();

            //----------------------Guizmo Creation------------------------------//
            void createObjectsGUIZMO();

            //----------------------Utils Methods--------------------------------//
            void initializeButtonImageTextures();

        public:
            Odysseus::Transform* transformToShow;
            
            Dockspace();
            
            void createDockspace();

    };
}

#endif // __DOCKSPACE_H__