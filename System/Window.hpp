#ifndef __ENGINE_WINDOW_H__
#define __ENGINE_WINDOW_H__

#include <imgui.h>
#include <imgui_internal.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <IO/Input.hpp>

#include <SceneGraph.hpp>
#include <Shader.hpp>
#include <Component.hpp>
#include <Transform.hpp>

#include <iostream>
#include <string>

namespace System {
    struct Screen {
        int width;
        int height;
    };

    class Window {    
        private:
            Odysseus::Transform* transformToShow;
            std::vector<Component*> inspectorParams;

            ImGuiID dockspace_id;

            Odysseus::Shader screenShader;
            
            GLuint framebuffer;
            GLuint screenVAO, screenVBO;
            GLuint rbo;
            GLuint textureColorbuffer;

            int sizeX, sizeY;

            void initializeImGUI();
            void createDockSpace();
            void initializeFrameBuffer();

        public:
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