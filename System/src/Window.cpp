#include "../Window.hpp"

namespace System {
    Buffers::FrameBuffer* Window::sceneFrameBuffer;
    Buffers::FrameBuffer* Window::gameFrameBuffer;

    GLFWwindow* Window::window;
    Screen Window::screen;

    bool Window::refreshFrameBuffer;

    Window::Window(std::string name, bool cursorDisabled)
    {
        screen.width = 1920;
        screen.height = 1080;

        if(!glfwInit())
            std::cerr << "Failed to initialize GLFW" << std::endl;
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 4);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        // TODO: Erase this
        glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GL_TRUE);

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
            Window::screen.width = w;
            Window::screen.height = h;

            if (Window::screen.width > Window::screen.height)
                glViewport(0, (Window::screen.height - Window::screen.width) / 2, Window::screen.width, Window::screen.width);
            else
                glViewport((Window::screen.width - Window::screen.height) / 2, 0, Window::screen.height, Window::screen.height);

            if (w < h) {
                Window::screen.width = h;
                Window::screen.height = h;
            } else {
                Window::screen.width = w;
                Window::screen.height = w;
            }
        };

        initializeMouseCallback();

        glfwMakeContextCurrent(this->window);
        glfwSetFramebufferSizeCallback(this->window, frameBufferCallback);

        if (cursorDisabled)
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        else
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        
        if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
            std::cerr << "Failed to initialize GLAD" << std::endl;
        
        initializeDebugCallback();

        // Enable depth test
        glEnable(GL_DEPTH_TEST);

        initializeImGUI();
        
        setWindowIcon();
    }

    Window::Window(const int& width, const int& height, std::string name, bool cursorDisabled)
    {
        screen.width = width;
        screen.height = height;

        glfwInit();
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 4);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        // TODO: Erease this
        glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GL_TRUE);

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
            Window::screen.width = w;
            Window::screen.height = h;

            if (Window::screen.width > Window::screen.height)
                glViewport(0, (Window::screen.height - Window::screen.width) / 2, Window::screen.width, Window::screen.width);
            else
                glViewport((Window::screen.width - Window::screen.height) / 2, 0, Window::screen.height, Window::screen.height);

            if (w < h) {
                Window::screen.width = h;
                Window::screen.height = h;
            } else {
                Window::screen.width = w;
                Window::screen.height = w;
            }
        };

        initializeMouseCallback();

        glfwMakeContextCurrent(this->window);
        glfwSetFramebufferSizeCallback(this->window, frameBufferCallback);

        if (cursorDisabled)
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        else
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);

        if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
            std::cerr << "Failed to initialize GLAD" << std::endl;

        initializeDebugCallback();

        // Enable depth test
        glEnable(GL_DEPTH_TEST);

        initializeImGUI();

        setWindowIcon();
    }

    void Window::initializeMouseCallback()
    {
        Input::mouse.isFirstMovement = true;
        Input::mouse.xPosition = screen.width / 2;
        Input::mouse.yPosition = screen.height / 2;
        Input::mouse.xPositionRelativeToSceneWindow = 0;
        Input::mouse.yPositionRelativeToSceneWindow = 0;
        Input::mouse.selectedObject = nullptr;

        auto mouseCallback = [](GLFWwindow *window, double xposIn, double yposIn) {
            float xpos = static_cast<float>(xposIn);
            float ypos = static_cast<float>(yposIn);

            if (Input::mouse.isFirstMovement)
            {
                Input::mouse.xPosition = xpos;
                Input::mouse.yPosition = ypos;
                Input::mouse.isFirstMovement = false;
            }

            Input::mouse.xPosition = xpos;
            Input::mouse.yPosition = ypos;
        };

        glfwSetCursorPosCallback(this->window, mouseCallback);
    }

    void Window::initializeDebugCallback()
    {
        glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);

        auto messageCallback = [](GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, const void* userParam) {
            if (severity == GL_DEBUG_SEVERITY_HIGH || severity == GL_DEBUG_SEVERITY_MEDIUM)
                std::cerr << std::endl << "[OpenGL Error](" << type << "): " << message << std::endl;
        };

        glDebugMessageCallback(messageCallback, 0);
    }

    void Window::setWindowIcon()
    {
        GLFWimage icons[1];
        int w, h, channels;
        stbi_uc* img = stbi_load("./Resources/gladiator.png", &w, &h, &channels, 0);
        icons->height = h;
        icons->width = w;
        icons[0].pixels = img;
        glfwSetWindowIcon(this->window, 1, icons);
    }

    void Window::initializeImGUI()
    {
        IMGUI_CHECKVERSION();

        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO();
        float fontSize = 15;
        io.Fonts->AddFontFromFileTTF(".\\Assets\\Fonts\\Roboto-Regular.ttf", fontSize);
        io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
        ImGui::StyleColorsDark();
        ImGui_ImplGlfw_InitForOpenGL(window, true);
        ImGui_ImplOpenGL3_Init("#version 410 core");
        ImGui::StyleColorsDark();

        const GLFWvidmode* mode = glfwGetVideoMode(glfwGetPrimaryMonitor());

        sceneFrameBuffer = new Buffers::FrameBuffer(mode->width * 2, mode->height * 2, true);

        Input::keyboard = new Keyboard();
    }

    bool Window::shouldWindowClose()
    {
        return glfwWindowShouldClose(this->window);
    }

    void Window::clear()
    {
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        ImGuizmo::BeginFrame();

        sceneFrameBuffer->bind();
    }

    void Window::update()
    {
        sceneFrameBuffer->blit();
        Odysseus::SceneManager::activeScene->sceneEditor->onEditorUpdate();

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        if (ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
        {
            GLFWwindow* backup_current_context = glfwGetCurrentContext();
            ImGui::UpdatePlatformWindows();
            ImGui::RenderPlatformWindowsDefault();
            glfwMakeContextCurrent(backup_current_context);
        }

        if (glfwGetKey(this->window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(this->window, true);

        glfwSwapBuffers(this->window);
        glfwPollEvents();
    }

    Window::~Window()
    {
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();

        glfwTerminate();
    }

}