#ifndef __DOCKSPACE_H__
#define __DOCKSPACE_H__

#include "Window.hpp"

#include <imgui.h>
#include <imgui_internal.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

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

    class Dockspace {
        private:
            Odysseus::Transform* transformToShow;

            std::vector<Component*> inspectorParams;

            Console* console;

            std::filesystem::path assetDirectory;
            std::filesystem::path currentDirectory;

            void createStyle();

            void createContentBrowser();
            void dfsOverFolders(std::filesystem::path sourceFolder, int index = 1);
            int countNestedFolders(std::filesystem::path sourceFolder);

        public:
            Dockspace();
            
            void createDockspace();

    };
}

#endif // __DOCKSPACE_H__