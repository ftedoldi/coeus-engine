#include "../Window.hpp"

namespace System {
    GLFWwindow* Window::window;
    Screen Window::screen;
    Screen Window::frameBufferSize;
    Odysseus::Shader* Window::screenShader;
    GLuint Window::textureColorbuffer;

    bool Window::refreshFrameBuffer;

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

        initializeImGUI();
        initializeQuad();
        initializeMSAAframebuffer();
        initializeFrameBuffer();
        
        setWindowIcon();
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

        initializeImGUI();
        initializeQuad();
        initializeMSAAframebuffer();
        initializeFrameBuffer();

        setWindowIcon();
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

    void Window::initializeQuad()
    {
        float screenVertices[] = {
        // positions   // texCoords
        -0.3f,  1.0f,  0.0f, 1.0f,
        -0.3f,  0.7f,  0.0f, 0.0f,
         0.3f,  0.7f,  1.0f, 0.0f,

        -0.3f,  1.0f,  0.0f, 1.0f,
         0.3f,  0.7f,  1.0f, 0.0f,
         0.3f,  1.0f,  1.0f, 1.0f
        };

        glGenVertexArrays(1, &screenVAO);
        glGenBuffers(1, &screenVBO);
        glBindVertexArray(screenVAO);
        glBindBuffer(GL_ARRAY_BUFFER, screenVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(screenVertices), &screenVertices, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    }

    void Window::initializeMSAAframebuffer()
    {
        // configure MSAA framebuffer
        // --------------------------
        glGenFramebuffers(1, &framebuffer);
        glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
        // create a multisampled color attachment texture
        glGenTextures(1, &textureColorBufferMultisample);
        glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, textureColorBufferMultisample);
        glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, 4, GL_RGB, frameBufferSize.width, frameBufferSize.height, GL_TRUE);
        glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, 0);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D_MULTISAMPLE, textureColorBufferMultisample, 0);
        // create a (also multisampled) renderbuffer object for depth and stencil attachments
        glGenRenderbuffers(1, &rbo);
        glBindRenderbuffer(GL_RENDERBUFFER, rbo);
        glRenderbufferStorageMultisample(GL_RENDERBUFFER, 4, GL_DEPTH24_STENCIL8, frameBufferSize.width, frameBufferSize.height);
        glBindRenderbuffer(GL_RENDERBUFFER, 0);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rbo);

        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
            std::cout << "ERROR::FRAMEBUFFER:: Framebuffer is not complete!" << std::endl;
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        refreshFrameBuffer = false;
    }

    void Window::initializeFrameBuffer()
    {
        // framebuffer configuration
        // -------------------------
        glGenFramebuffers(1, &intermediateFBO);
        glBindFramebuffer(GL_FRAMEBUFFER, intermediateFBO);

        // create a color attachment texture
        glGenTextures(1, &textureColorbuffer);
        glBindTexture(GL_TEXTURE_2D, textureColorbuffer);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, frameBufferSize.width, frameBufferSize.height, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textureColorbuffer, 0);

        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
            std::cout << "ERROR::FRAMEBUFFER:: Framebuffer is not complete!" << std::endl;
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        screenShader = new Odysseus::Shader(".\\Shader\\frameBufferShader.vert", ".\\Shader\\frameBufferShader.frag");

        screenShader->use();
        screenShader->setInt("screenTexture", 0);

        refreshFrameBuffer = false;
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

        frameBufferSize.width = mode->width * 2;
        frameBufferSize.height = mode->height * 2;

        dockspace = new Dockspace();

        Input::keyboard = new Keyboard();
    }

    bool Window::shouldWindowClose()
    {
        return glfwWindowShouldClose(this->window);
    }

    void Window::resetFrameBufferTexture()
    {
        glBindTexture(GL_TEXTURE_2D, textureColorbuffer);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, frameBufferSize.width, frameBufferSize.height, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
    }

    void Window::clear()
    {
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        ImGuizmo::BeginFrame();

        if (refreshFrameBuffer) 
        {
            resetFrameBufferTexture();
            refreshFrameBuffer = false;
        }
        
        glfwGetFramebufferSize(window, &sizeX, &sizeY);

        glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
        glViewport(0, 0, frameBufferSize.width, frameBufferSize.height);
        glEnable(GL_DEPTH_TEST); // enable depth testing (is disabled for rendering screen-space quad)

        glClearColor(0.4f, 0.2f, 0.6f, 0.5f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    }

    void Window::update()
    {
        glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, intermediateFBO);
        glBlitFramebuffer(0, 0, frameBufferSize.width, frameBufferSize.height, 0, 0, frameBufferSize.width, frameBufferSize.height, GL_COLOR_BUFFER_BIT, GL_NEAREST);

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        // glViewport(0, 0, frameBufferSize.width, frameBufferSize.height);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f); 
        glClear(GL_COLOR_BUFFER_BIT);
        
        glDisable(GL_DEPTH_TEST); // disable depth test so screen-space quad isn't discarded due to depth test.
        
        screenShader->use();
        glBindVertexArray(screenVAO);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, textureColorbuffer);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        dockspace->createDockspace();

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

    void Window::framebufferShaderCallback(const ImDrawList* asd, const ImDrawCmd* command)
    {
        ImDrawData* draw_data = ImGui::GetDrawData();
        float L = draw_data->DisplayPos.x;
        float R = draw_data->DisplayPos.x + draw_data->DisplaySize.x;
        float T = draw_data->DisplayPos.y;
        float B = draw_data->DisplayPos.y + draw_data->DisplaySize.y;

        const float ortho_projection[4][4] =
        {
            { 2.0f/(R-L),   0.0f,         0.0f,   0.0f },
            { 0.0f,         2.0f/(T-B),   0.0f,   0.0f },
            { 0.0f,         0.0f,        -1.0f,   0.0f },
            { (R+L)/(L-R),  (T+B)/(B-T),  0.0f,   1.0f },
        };

        Athena::Matrix4 projection(ortho_projection[0][0],ortho_projection[0][1], ortho_projection[0][2], ortho_projection[0][3],
                                   ortho_projection[1][0],ortho_projection[1][1], ortho_projection[1][2], ortho_projection[1][3], 
                                   ortho_projection[2][0],ortho_projection[2][1], ortho_projection[2][2], ortho_projection[2][3],
                                   ortho_projection[3][0],ortho_projection[3][1], ortho_projection[3][2], ortho_projection[3][3]
                                );

        Window::screenShader->use();
        Window::screenShader->setMat4("projection", projection);
    }

    Window::~Window()
    {
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();

        glfwTerminate();
    }

}