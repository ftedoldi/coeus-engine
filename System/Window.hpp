#ifndef __ENGINE_WINDOW_H__
#define __ENGINE_WINDOW_H__

#include <imgui.h>
#include <imgui_internal.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>

#include <IO/Input.hpp>

#include <SceneGraph.hpp>
#include <Shader.hpp>
#include <Component.hpp>
#include <Transform.hpp>
#include <Math.hpp>

#include "Debug.hpp"

#include <iostream>
#include <string>
#include <cmath>

namespace System {
    struct Screen {
        int width;
        int height;
    };

    class Component;
    class Console;

    class Window {    
        private:
            Odysseus::Transform* transformToShow;
            std::vector<Component*> inspectorParams;

            ImGuiID dockspace_id;

            GLuint framebuffer;
            GLuint screenVAO, screenVBO;
            GLuint rbo;
            GLuint textureColorbuffer;
            GLuint texture;

            int sizeX, sizeY;

            Console* console;

            void initializeImGUI();
            void createDockSpace();
            void initializeFrameBuffer();
            static void useShader(const ImDrawList*, const ImDrawCmd* command);

        public:
            static Odysseus::Shader* screenShader;

            static GLFWwindow* window;
            static Screen screen;

            Window(std::string name = "MyApplication", bool cursorDisabled = false);
            Window(const int& width, const int& height, std::string name = "MyApplication", bool cursorDisabled = false);

            bool shouldWindowClose();

            void clear();
            void update();

            ~Window();
    };
}

#endif // __ENGINE_WINDOW_H__