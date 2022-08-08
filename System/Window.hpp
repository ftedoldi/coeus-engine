#ifndef __ENGINE_WINDOW_H__
#define __ENGINE_WINDOW_H__

#include <imgui.h>
#include <imgui_internal.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <ImGuizmo.h>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>

#include <IO/Input.hpp>
#include <Folder.hpp>
#include <Buffers/FrameBuffer.hpp>
#include <PixelData.hpp>

#include <SceneGraph.hpp>
#include <Shader.hpp>
#include <Component.hpp>
#include <Transform.hpp>
#include <Texture2D.hpp>
#include <Math.hpp>

#include "Dockspace.hpp"

#include <iostream>
#include <string>
#include <cmath>
#include <filesystem>
#include <vector>
#include <algorithm>
#include <random>

#include <stb/stb_image.h>

namespace System {
    struct Screen {
        int width;
        int height;
    };

    class Component;
    class Dockspace;

    class Window {
        friend class Dockspace;

        private:
            Dockspace* dockspace;
            
            void setWindowIcon();
            void initializeImGUI();

        public:
            static GLFWwindow* window;
            static Screen screen;
            
            static bool refreshFrameBuffer;

            static Buffers::FrameBuffer* sceneFrameBuffer;
            static Buffers::FrameBuffer* gameFrameBuffer;

            Window(std::string name = "MyApplication", bool cursorDisabled = false);
            Window(const int& width, const int& height, std::string name = "MyApplication", bool cursorDisabled = false);
            
            bool shouldWindowClose();

            void clear();
            void update();

            void bindGameFrameBuffer();
            void updateGameFrameBuffer();

            ~Window();
    };
}

#endif // __ENGINE_WINDOW_H__