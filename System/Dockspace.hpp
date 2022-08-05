#ifndef __DOCKSPACE_H__
#define __DOCKSPACE_H__

#include "Window.hpp"

#include <imgui.h>
#include <imgui_internal.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <ImGuizmo.h>

#include <IO/Input.hpp>
#include <Folder.hpp>
#include <StatusBar.hpp>

#include "Debug.hpp"

#include <SceneGraph.hpp>
#include <Shader.hpp>
#include <SceneGraph.hpp>
#include <Transform.hpp>
#include <Component.hpp>
#include <Texture2D.hpp>
#include <Math.hpp>
#include <Camera.hpp>
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
    class Odysseus::Transform;
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
    };

    class Dockspace {
        private:
            Odysseus::Transform* transformToShow;

            std::vector<Component*> inspectorParams;

            Console* console;

            StatusBar* statusBar;

            std::filesystem::path assetDirectory;
            std::filesystem::path currentDirectory;

            ButtonImages buttonImages;

            ImGuizmo::OPERATION gizmoOperation;

            void createStyle();

            //----------------------Menu Creation---------------------------------//
            void createMainMenuBar();
            void createToolMenuBar();
            void createStatusMenuBar();

            //----------------------Window Creation-------------------------------//
            void createHierarchyWindow();
            void createInspectorWindow();
            void createConsoleWindow();
            void createContentBrowser();
            void createSceneWindow();
            void createGameWindow();
            void createProjectSettingsWindow();

            //----------------------Utils Methods--------------------------------//
            void dfsOverFolders(std::filesystem::path sourceFolder, int index = 1);
            int countNestedFolders(std::filesystem::path sourceFolder);
            void dfsOverChildren(Odysseus::Transform* childrenTransform, int index = 1);
            int countNestedChildren(Odysseus::Transform* childrenTransform);

            void initializeButtonImageTextures();

        public:
            Dockspace();
            
            void createDockspace();

    };
}

#endif // __DOCKSPACE_H__