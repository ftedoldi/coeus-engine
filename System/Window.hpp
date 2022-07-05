#ifndef __ENGINE_WINDOW_H__
#define __ENGINE_WINDOW_H__

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <IO/Input.hpp>

#include <iostream>
#include <string>

namespace System {
    struct Screen {
        int width;
        int height;
    };

    class Window {         
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