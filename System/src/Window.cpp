#include "../Window.hpp"

namespace System {
    GLFWwindow* Window::window;
    Screen Window::screen;

    Window::Window(std::string name, bool cursorDisabled)
    {
        screen.width = 800;
        screen.height = 600;

        if(!glfwInit())
            std::cerr << "Failed to initialize GLFW" << std::endl;
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

        #ifdef __APPLE__
            glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
        #endif

        this->window = glfwCreateWindow(screen.width, screen.height, name.c_str(), NULL, NULL);

        if (window == NULL)
        {
            std::cerr << "Failed to create GLFW window" << std::endl;
            glfwTerminate();
        }

        auto frameBufferCallback = [](GLFWwindow* window, int w, int h) {
            glViewport(0, 0, w, h);
        };

        Input::mouse.isFirstMovement = true;
        Input::mouse.xPosition = screen.width / 2;
        Input::mouse.yPosition = screen.height / 2;

        auto mouseCallback = [](GLFWwindow *window, double xposIn, double yposIn) {
            float xpos = static_cast<Athena::Scalar>(xposIn);
            float ypos = static_cast<Athena::Scalar>(yposIn);

            if (Input::mouse.isFirstMovement)
            {
                Input::mouse.xPosition = xpos;
                Input::mouse.yPosition = ypos;
                Input::mouse.isFirstMovement = false;
            }

            Input::mouse.xOffsetFromLastPosition = xpos - Input::mouse.xPosition;
            Input::mouse.yOffsetFromLastPosition = Input::mouse.yPosition - ypos; // reversed since y-coordinates go from bottom to top

            Input::mouse.xPosition = xpos;
            Input::mouse.yPosition = ypos;
        };

        glfwMakeContextCurrent(this->window);
        glfwSetFramebufferSizeCallback(this->window, frameBufferCallback);
        glfwSetCursorPosCallback(this->window, mouseCallback);

        if (cursorDisabled)
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        else
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        
        if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
            std::cerr << "Failed to initialize GLAD" << std::endl;
        
        // Enable depth test
        glEnable(GL_DEPTH_TEST);
    }

    Window::Window(const int& width, const int& height, std::string name, bool cursorDisabled)
    {
        screen.width = width;
        screen.height = height;

        glfwInit();
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

        #ifdef __APPLE__
            glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
        #endif

        this->window = glfwCreateWindow(screen.width, screen.height, name.c_str(), NULL, NULL);

        if (window == NULL)
        {
            std::cerr << "Failed to create GLFW window" << std::endl;
            glfwTerminate();
        }

        auto frameBufferCallback = [](GLFWwindow* window, int w, int h) {
            glViewport(0, 0, w, h);
        };

        Input::mouse.isFirstMovement = true;
        Input::mouse.xPosition = screen.width / 2;
        Input::mouse.yPosition = screen.height / 2;

        auto mouseCallback = [](GLFWwindow *window, double xposIn, double yposIn) {
            float xpos = static_cast<float>(xposIn);
            float ypos = static_cast<float>(yposIn);

            if (Input::mouse.isFirstMovement)
            {
                Input::mouse.xPosition = xpos;
                Input::mouse.yPosition = ypos;
                Input::mouse.isFirstMovement = false;
            }

            Input::mouse.xOffsetFromLastPosition = xpos - Input::mouse.xPosition;
            Input::mouse.yOffsetFromLastPosition = Input::mouse.yPosition - ypos; // reversed since y-coordinates go from bottom to top

            Input::mouse.xPosition = xpos;
            Input::mouse.yPosition = ypos;
        };

        glfwMakeContextCurrent(this->window);
        glfwSetFramebufferSizeCallback(this->window, frameBufferCallback);
        glfwSetCursorPosCallback(this->window, mouseCallback);

        if (cursorDisabled)
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        else
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);

        if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
            std::cerr << "Failed to initialize GLAD" << std::endl;

        // Enable depth test
        glEnable(GL_DEPTH_TEST);
    }

    bool Window::shouldWindowClose()
    {
        return glfwWindowShouldClose(this->window);
    }

    void Window::clear()
    {
        glClearColor(0.4f, 0.2f, 0.6f, 0.5f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    }

    void Window::update()
    {
        if (glfwGetKey(this->window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(this->window, true);

        glfwSwapBuffers(this->window);
        glfwPollEvents();
    }

    Window::~Window()
    {
        glfwTerminate();
    }

}