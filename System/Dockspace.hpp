#ifndef __DOCKSPACE_H__
#define __DOCKSPACE_H__

#include "Window.hpp"

#include <imgui.h>
#include <imgui_internal.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <glm/glm.hpp>
#include <glm/common.hpp>
#include <glm/mat4x4.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/matrix_decompose.hpp>

#include <ImGuizmo.h>

#include <IO/Input.hpp>
#include <Folder.hpp>

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

    enum TextColor {
        WHITE,
        RED,
        YELLOW,
        GREEN
    };

    struct CurrentStatus {
        std::string statusText;
        TextColor statusTextColor;
    };

    class Dockspace {
        private:
            Odysseus::Transform* transformToShow;

            std::vector<Component*> inspectorParams;

            Console* console;

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