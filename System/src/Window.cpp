#include "../Window.hpp"

namespace System {
    GLFWwindow* Window::window;
    Screen Window::screen;
    Odysseus::Shader* Window::screenShader;

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
        glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, 4, GL_RGB, System::Window::screen.width, System::Window::screen.height, GL_TRUE);
        glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, 0);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D_MULTISAMPLE, textureColorBufferMultisample, 0);
        // create a (also multisampled) renderbuffer object for depth and stencil attachments
        glGenRenderbuffers(1, &rbo);
        glBindRenderbuffer(GL_RENDERBUFFER, rbo);
        glRenderbufferStorageMultisample(GL_RENDERBUFFER, 4, GL_DEPTH24_STENCIL8, System::Window::screen.width, System::Window::screen.height);
        glBindRenderbuffer(GL_RENDERBUFFER, 0);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rbo);

        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
            std::cout << "ERROR::FRAMEBUFFER:: Framebuffer is not complete!" << std::endl;
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
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
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, System::Window::screen.width, System::Window::screen.height, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textureColorbuffer, 0);

        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
            std::cout << "ERROR::FRAMEBUFFER:: Framebuffer is not complete!" << std::endl;
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        screenShader = new Odysseus::Shader(".\\Shader\\frameBufferShader.vert", ".\\Shader\\frameBufferShader.frag");

        screenShader->use();
        screenShader->setInt("screenTexture", 0);
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

        this->transformToShow = nullptr;

        this->console = new Console();
        Debug::mainConsole = this->console;

        this->assetDirectory = Folder::getFolderPath("Assets");
        this->currentDirectory = this->assetDirectory;
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

        glfwGetFramebufferSize(window, &sizeX, &sizeY);

        glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
        glViewport(0, 0, System::Window::screen.width, System::Window::screen.height);
        glEnable(GL_DEPTH_TEST); // enable depth testing (is disabled for rendering screen-space quad)

        glClearColor(0.4f, 0.2f, 0.6f, 0.5f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    }

    void Window::update()
    {
        glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, intermediateFBO);
        glBlitFramebuffer(0, 0, System::Window::screen.width, System::Window::screen.height, 0, 0, System::Window::screen.width, System::Window::screen.height, GL_COLOR_BUFFER_BIT, GL_NEAREST);

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glViewport(0, 0, sizeX, sizeY);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f); 
        glClear(GL_COLOR_BUFFER_BIT);
        
        glDisable(GL_DEPTH_TEST); // disable depth test so screen-space quad isn't discarded due to depth test.
        
        screenShader->use();
        glBindVertexArray(screenVAO);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, textureColorbuffer);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        createDockSpace();

        ImGui::Render();
        // int display_w, display_h;
        // glfwGetFramebufferSize(window, &display_w, &display_h);
        // glViewport(0, 0, display_w, display_h);
        // glClearColor(1.0f, 1.0f, 1.0f, 1.0f); 
        // glClear(GL_COLOR_BUFFER_BIT);
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

    void Window::createDockSpace() 
    {
        auto ColorFromBytes = [](uint8_t r, uint8_t g, uint8_t b)
        {
            return ImVec4((float)r / 255.0f, (float)g / 255.0f, (float)b / 255.0f, 1.0f);
        };

        auto& style = ImGui::GetStyle();

        style.Colors[ImGuiCol_Text]                  = ImVec4(1.00f, 1.00f, 1.00f, 1.00f);
        style.Colors[ImGuiCol_TextDisabled]          = ImVec4(0.50f, 0.50f, 0.50f, 1.00f);
        style.Colors[ImGuiCol_WindowBg]              = ImVec4(0.13f, 0.14f, 0.15f, 1.00f);
        style.Colors[ImGuiCol_ChildBg]               = ImVec4(0.13f, 0.14f, 0.15f, 1.00f);
        style.Colors[ImGuiCol_PopupBg]               = ImVec4(0.13f, 0.14f, 0.15f, 1.00f);
        style.Colors[ImGuiCol_Border]                = ImVec4(0.43f, 0.43f, 0.50f, 0.50f);
        style.Colors[ImGuiCol_BorderShadow]          = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
        style.Colors[ImGuiCol_FrameBg]               = ImVec4(0.25f, 0.25f, 0.25f, 1.00f);
        style.Colors[ImGuiCol_FrameBgHovered]        = ImVec4(0.38f, 0.38f, 0.38f, 1.00f);
        style.Colors[ImGuiCol_FrameBgActive]         = ImVec4(0.67f, 0.67f, 0.67f, 0.39f);
        style.Colors[ImGuiCol_TitleBg]               = ImVec4(0.08f, 0.08f, 0.09f, 1.00f);
        style.Colors[ImGuiCol_TitleBgActive]         = ImVec4(0.08f, 0.08f, 0.09f, 1.00f);
        style.Colors[ImGuiCol_TitleBgCollapsed]      = ImVec4(0.00f, 0.00f, 0.00f, 0.51f);
        style.Colors[ImGuiCol_MenuBarBg]             = ImVec4(0.14f, 0.14f, 0.14f, 1.00f);
        style.Colors[ImGuiCol_ScrollbarBg]           = ImVec4(0.02f, 0.02f, 0.02f, 0.53f);
        style.Colors[ImGuiCol_ScrollbarGrab]         = ImVec4(0.31f, 0.31f, 0.31f, 1.00f);
        style.Colors[ImGuiCol_ScrollbarGrabHovered]  = ImVec4(0.41f, 0.41f, 0.41f, 1.00f);
        style.Colors[ImGuiCol_ScrollbarGrabActive]   = ImVec4(0.51f, 0.51f, 0.51f, 1.00f);
        style.Colors[ImGuiCol_CheckMark]             = ImVec4(0.11f, 0.64f, 0.92f, 1.00f);
        style.Colors[ImGuiCol_SliderGrab]            = ImVec4(0.11f, 0.64f, 0.92f, 1.00f);
        style.Colors[ImGuiCol_SliderGrabActive]      = ImVec4(0.08f, 0.50f, 0.72f, 1.00f);
        style.Colors[ImGuiCol_Button]                = ImVec4(0.25f, 0.25f, 0.25f, 1.00f);
        style.Colors[ImGuiCol_ButtonHovered]         = ImVec4(0.38f, 0.38f, 0.38f, 1.00f);
        style.Colors[ImGuiCol_ButtonActive]          = ImVec4(0.67f, 0.67f, 0.67f, 0.39f);
        style.Colors[ImGuiCol_Header]                = ColorFromBytes(121, 170, 247);
        style.Colors[ImGuiCol_HeaderHovered]         = ColorFromBytes(199, 220, 252);
        style.Colors[ImGuiCol_HeaderActive]          = ColorFromBytes(121, 170, 247);
        style.Colors[ImGuiCol_Separator]             = style.Colors[ImGuiCol_Border];
        style.Colors[ImGuiCol_SeparatorHovered]      = ImVec4(0.41f, 0.42f, 0.44f, 1.00f);
        style.Colors[ImGuiCol_SeparatorActive]       = ImVec4(0.26f, 0.59f, 0.98f, 0.95f);
        style.Colors[ImGuiCol_ResizeGrip]            = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
        style.Colors[ImGuiCol_ResizeGripHovered]     = ImVec4(0.29f, 0.30f, 0.31f, 0.67f);
        style.Colors[ImGuiCol_ResizeGripActive]      = ImVec4(0.26f, 0.59f, 0.98f, 0.95f);
        style.Colors[ImGuiCol_Tab]                   = ImVec4(0.08f, 0.08f, 0.09f, 0.83f);
        style.Colors[ImGuiCol_TabHovered]            = ImVec4(0.33f, 0.34f, 0.36f, 0.83f);
        style.Colors[ImGuiCol_TabActive]             = ImVec4(0.23f, 0.23f, 0.24f, 1.00f);
        style.Colors[ImGuiCol_TabUnfocused]          = ImVec4(0.08f, 0.08f, 0.09f, 1.00f);
        style.Colors[ImGuiCol_TabUnfocusedActive]    = ImVec4(0.13f, 0.14f, 0.15f, 1.00f);
        style.Colors[ImGuiCol_DockingPreview]        = ImVec4(0.26f, 0.59f, 0.98f, 0.70f);
        style.Colors[ImGuiCol_DockingEmptyBg]        = ImVec4(0.20f, 0.20f, 0.20f, 1.00f);
        style.Colors[ImGuiCol_PlotLines]             = ImVec4(0.61f, 0.61f, 0.61f, 1.00f);
        style.Colors[ImGuiCol_PlotLinesHovered]      = ImVec4(1.00f, 0.43f, 0.35f, 1.00f);
        style.Colors[ImGuiCol_PlotHistogram]         = ImVec4(0.90f, 0.70f, 0.00f, 1.00f);
        style.Colors[ImGuiCol_PlotHistogramHovered]  = ImVec4(1.00f, 0.60f, 0.00f, 1.00f);
        style.Colors[ImGuiCol_TextSelectedBg]        = ImVec4(0.26f, 0.59f, 0.98f, 0.35f);
        style.Colors[ImGuiCol_DragDropTarget]        = ImVec4(0.11f, 0.64f, 0.92f, 1.00f);
        style.Colors[ImGuiCol_NavHighlight]          = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
        style.Colors[ImGuiCol_NavWindowingHighlight] = ImVec4(1.00f, 1.00f, 1.00f, 0.70f);
        style.Colors[ImGuiCol_NavWindowingDimBg]     = ImVec4(0.80f, 0.80f, 0.80f, 0.20f);
        style.Colors[ImGuiCol_ModalWindowDimBg]      = ImVec4(0.80f, 0.80f, 0.80f, 0.35f);
        style.Colors[ImGuiCol_Header]                = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
        style.Colors[ImGuiCol_HeaderHovered]         = ImVec4(0.38f, 0.38f, 0.38f, 1.00f);
        style.Colors[ImGuiCol_HeaderActive]          = ImVec4(0.67f, 0.67f, 0.67f, 0.39f);

        style.WindowRounding    = 2.0f;
        style.ChildRounding     = 2.0f;
        style.FrameRounding     = 2.0f;
        style.GrabRounding      = 2.0f;
        style.PopupRounding     = 2.0f;
        style.ScrollbarRounding = 2.0f;
        style.TabRounding       = 2.0f;

        static bool opt_fullscreen = true;
        static bool opt_padding = false;
        static ImGuiDockNodeFlags dockspace_flags = ImGuiDockNodeFlags_NoCloseButton;
        bool* p_open = new bool;
        *p_open = true;

        // We are using the ImGuiWindowFlags_NoDocking flag to make the parent window not dockable into,
        // because it would be confusing to have two docking targets within each others.
        ImGuiWindowFlags window_flags = ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoDocking;
        if (opt_fullscreen)
        {
            const ImGuiViewport* viewport = ImGui::GetMainViewport();
            ImGui::SetNextWindowPos(viewport->WorkPos);
            ImGui::SetNextWindowSize(viewport->WorkSize);
            ImGui::SetNextWindowViewport(viewport->ID);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
            window_flags |= ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove;
            window_flags |= ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;
        }
        else
        {
            dockspace_flags &= ~ImGuiDockNodeFlags_PassthruCentralNode;
        }

        // When using ImGuiDockNodeFlags_PassthruCentralNode, DockSpace() will render our background
        // and handle the pass-thru hole, so we ask Begin() to not render a background.
        if (dockspace_flags & ImGuiDockNodeFlags_PassthruCentralNode)
            window_flags |= ImGuiWindowFlags_NoBackground;

        // Important: note that we proceed even if Begin() returns false (aka window is collapsed).
        // This is because we want to keep our DockSpace() active. If a DockSpace() is inactive,
        // all active windows docked into it will lose their parent and become undocked.
        // We cannot preserve the docking relationship between an active window and an inactive docking, otherwise
        // any change of dockspace/settings would lead to windows being stuck in limbo and never being visible.
        if (!opt_padding)
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
        ImGui::Begin("MyDockSpace", p_open, window_flags);
        if (!opt_padding)
            ImGui::PopStyleVar();

        if (opt_fullscreen)
            ImGui::PopStyleVar(2);

        // Submit the DockSpace
        ImGuiIO& io = ImGui::GetIO();
        this->dockspace_id = ImGui::GetID("MyDockSpace");
        ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), dockspace_flags);

        static auto first_time = true;
	    if (first_time)
	    {
	    	first_time = false;

	    	ImGui::DockBuilderRemoveNode(dockspace_id); // clear any previous layout
	    	ImGui::DockBuilderAddNode(dockspace_id, dockspace_flags | ImGuiDockNodeFlags_DockSpace);
	    	ImGui::DockBuilderSetNodeSize(dockspace_id, ImVec2(screen.width, screen.height));

	    	// split the dockspace into 2 nodes -- DockBuilderSplitNode takes in the following args in the following order
	    	//   window ID to split, direction, fraction (between 0 and 1), the final two setting let's us choose which id we want (which ever one we DON'T set as NULL, will be returned by the function)
	    	//                                                              out_id_at_dir is the id of the node in the direction we specified earlier, out_id_at_opposite_dir is in the opposite direction
	    	auto dock_id_down_left = ImGui::DockBuilderSplitNode(dockspace_id, ImGuiDir_Down, 0.3f, nullptr, &dockspace_id);
	    	auto dock_id_down_right = ImGui::DockBuilderSplitNode(dock_id_down_left, ImGuiDir_Right, 0.3f, nullptr, &dock_id_down_left);
	    	auto dock_id_left = ImGui::DockBuilderSplitNode(dockspace_id, ImGuiDir_Left, 0.2f, nullptr, &dockspace_id);
	    	auto dock_id_right = ImGui::DockBuilderSplitNode(dockspace_id, ImGuiDir_Right, 0.2f, nullptr, &dockspace_id);

	    	// we now dock our windows into the docking node we made above
	    	ImGui::DockBuilderDockWindow("Project", dock_id_down_left);
	    	ImGui::DockBuilderDockWindow("Console", dock_id_down_left);
	    	ImGui::DockBuilderDockWindow("Game", dock_id_down_right);
	    	ImGui::DockBuilderDockWindow("Project Settings", dock_id_down_right);
	    	ImGui::DockBuilderDockWindow("Hierarchy", dock_id_left);
	    	ImGui::DockBuilderDockWindow("Inspector", dock_id_right);
	    	ImGui::DockBuilderDockWindow("Scene", dockspace_id);
	    	ImGui::DockBuilderFinish(dockspace_id);
	    }

        ImGui::PushStyleColor(ImGuiCol_MenuBarBg, ImVec4(1.0f, 1.0f, 1.0f, 1.0f)); // Menu bar background color
        ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(0,0,0,255));
        ImGui::PushStyleColor(ImGuiCol_Border, IM_COL32(255,255,255,255));
        if (ImGui::BeginViewportSideBar("Main Menu Bar", ImGui::GetMainViewport(), ImGuiDir_Up, ImGui::GetFrameHeight(), ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_MenuBar))
        {
            if (ImGui::BeginMenuBar())
            {
                ImGui::PushStyleColor(ImGuiCol_PopupBg, ImVec4(1.0f, 1.0f, 1.0f, 1.0f)); // Menu bar background color
                if (ImGui::BeginMenu("Options"))
                {
                    ImGui::MenuItem("Padding", NULL, &opt_padding);
                    ImGui::Separator();

                    if (ImGui::MenuItem("Flag: NoSplit",                "", (dockspace_flags & ImGuiDockNodeFlags_NoSplit) != 0))                 
                        dockspace_flags ^= ImGuiDockNodeFlags_NoSplit;

                    if (ImGui::MenuItem("Flag: NoResize",               "", (dockspace_flags & ImGuiDockNodeFlags_NoResize) != 0))                
                        dockspace_flags ^= ImGuiDockNodeFlags_NoResize;

                    if (ImGui::MenuItem("Flag: NoDockingInCentralNode", "", (dockspace_flags & ImGuiDockNodeFlags_NoDockingInCentralNode) != 0))  
                        dockspace_flags ^= ImGuiDockNodeFlags_NoDockingInCentralNode;

                    if (ImGui::MenuItem("Flag: AutoHideTabBar",         "", (dockspace_flags & ImGuiDockNodeFlags_AutoHideTabBar) != 0))          
                        dockspace_flags ^= ImGuiDockNodeFlags_AutoHideTabBar;

                    if (ImGui::MenuItem("Flag: PassthruCentralNode",    "", (dockspace_flags & ImGuiDockNodeFlags_PassthruCentralNode) != 0, opt_fullscreen)) 
                        dockspace_flags ^= ImGuiDockNodeFlags_PassthruCentralNode;

                    ImGui::Separator();

                    if (ImGui::MenuItem("Close", NULL, false, p_open != NULL))
                        *p_open = false;
                    ImGui::EndMenu();
                }
                ImGui::PopStyleColor();
                ImGui::EndMenuBar();
            }
            
            ImGui::End();
        }
        ImGui::PopStyleColor(3);

        ImGui::PushStyleColor(ImGuiCol_Border, IM_COL32(255,255,255,0));
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, {0, 2});
        if (ImGui::BeginViewportSideBar("Tool Bar", ImGui::GetMainViewport(), ImGuiDir_Up, 1, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_MenuBar))
        {
            if (ImGui::BeginMenuBar()) {
                ImGui::Text("Happy tool bar");
                ImGui::EndMenuBar();
            }
            ImGui::End();
        }
        ImGui::PopStyleColor();
        ImGui::PopStyleVar();

        ImGui::PushStyleColor(ImGuiCol_Border, IM_COL32(255,255,255,0));
        if (ImGui::BeginViewportSideBar("Status Bar", ImGui::GetMainViewport(), ImGuiDir_Down, ImGui::GetFrameHeight(), ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_MenuBar))
        {
            if (ImGui::BeginMenuBar()) {
                ImGui::Text("Happy status bar");
                ImGui::EndMenuBar();
            }
            ImGui::End();
        }
        ImGui::PopStyleColor();

        ImGui::End();

        static Odysseus::Transform* lastTransform = nullptr;
        ImGui::Begin("Hierarchy");
            for (int i = 0; i < Odysseus::SceneGraph::objectsInScene.size(); i++) {
                if (ImGui::Button(Odysseus::SceneGraph::objectsInScene[i]->transform->name.c_str())) {
                    this->transformToShow = Odysseus::SceneGraph::objectsInScene[i]->transform;
                    this->inspectorParams.clear();
                    for (int j = 0; j < Odysseus::SceneGraph::objectsInScene[i]->_container->components.size(); j++)
                        this->inspectorParams.push_back(Odysseus::SceneGraph::objectsInScene[i]->_container->components[j]);
                }
            }
        ImGui::End();

        static bool isOpen = true;

        console->Draw("Console", &isOpen);
        // TODO: find texture icons and add these icons over textures
        createContentBrowser();

        ImGui::Begin("Inspector");
            if (this->transformToShow != nullptr) {
                ImGui::Text("Transform");
                float pos[] = { this->transformToShow->position.coordinates.x, this->transformToShow->position.coordinates.y, this->transformToShow->position.coordinates.z };
                ImGui::InputFloat3("Position", pos);
                this->transformToShow->position = Athena::Vector3(pos[0], pos[1], pos[2]);

                static bool firstRotation = true;
                static float rotation[3];
                // TODO: Fix this in order to prevent rounding errors of toEulerAnglesMethod
                if (lastTransform != transformToShow) {
                    Athena::Vector3 rot(this->transformToShow->rotation.toEulerAngles());
                    rotation[0] = rot.coordinates.x;
                    rotation[1] = rot.coordinates.y;
                    rotation[2] = rot.coordinates.z;
                    firstRotation = false;
                    lastTransform = transformToShow;
                }
                ImGui::InputFloat3("Rotation", rotation);
                this->transformToShow->rotation = Athena::Quaternion(
                                                                        0, 
                                                                        0,
                                                                        std::sin(Athena::Math::degreeToRandiansAngle(rotation[2])/2),
                                                                        std::cos(Athena::Math::degreeToRandiansAngle(rotation[2])/2)
                                                                    )
                                                * Athena::Quaternion(
                                                                        0, 
                                                                        std::sin(Athena::Math::degreeToRandiansAngle(rotation[1])/2), 
                                                                        0, 
                                                                        std::cos(Athena::Math::degreeToRandiansAngle(rotation[1])/2)
                                                                    )
                                                * Athena::Quaternion(
                                                                        std::sin(Athena::Math::degreeToRandiansAngle(rotation[0])/2),
                                                                        0, 
                                                                        0, 
                                                                        std::cos(Athena::Math::degreeToRandiansAngle(rotation[0])/2)
                                                                    );

                float scale[] = { this->transformToShow->localScale.coordinates.x, this->transformToShow->localScale.coordinates.y, this->transformToShow->localScale.coordinates.z };
                ImGui::InputFloat3("Scale", scale);
                this->transformToShow->localScale = Athena::Vector3(scale[0], scale[1], scale[2]);

                ImGui::Separator();
                for (int i = 0; i < this->inspectorParams.size(); i++) {
                    ImGui::Text(this->inspectorParams[i]->toString().c_str());
                    ImGui::Separator();
                }
            }
        ImGui::End();

        ImGui::Begin("Scene");
            ImGui::BeginChild("Game Render");
                ImDrawList* drawList = ImGui::GetWindowDrawList();
                drawList->AddCallback(&framebufferShaderCallback, nullptr);
                ImVec2 wSize = ImGui::GetWindowSize();

                ImGuiWindow* w = ImGui::GetCurrentWindow();

                ImGui::Image((ImTextureID)textureColorbuffer, wSize, ImVec2(0, 1), ImVec2(1, 0));
                drawList->AddCallback(ImDrawCallback_ResetRenderState, nullptr);
            ImGui::EndChild();
        ImGui::End();

        ImGui::Begin("Project Settings");
        ImGui::Text("Hello, right!");
        ImGui::End();

        // TODO: setup a frame buffer for the Game Scene
        ImGui::Begin("Game");
            ImGui::BeginChild("Game Render");
                ImDrawList* dList = ImGui::GetWindowDrawList();
                dList->AddCallback(&framebufferShaderCallback, nullptr);
                ImVec2 size = ImGui::GetWindowSize();

                ImGui::Image((ImTextureID)textureColorbuffer, size, ImVec2(0, 1), ImVec2(1, 0));
                dList->AddCallback(ImDrawCallback_ResetRenderState, nullptr);
            ImGui::EndChild();
        ImGui::End();
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

    // TODO: Fix forward button bug
    void Window::createContentBrowser()
    {
        static int actionIndex = 0;
        static std::vector<std::filesystem::path> actions = { this->assetDirectory };

        static ImGuiTextFilter filter;
        
        ImGui::Begin("Project", NULL, ImGuiWindowFlags_NoScrollbar);
            ImGui::Columns(2);

            static bool isFirstOpening = true;
            if (isFirstOpening) 
            {
                ImGui::SetColumnWidth(0, ImGui::GetContentRegionAvail().x / 1.8);
                isFirstOpening = false;
            }

            if(ImGui::CollapsingHeader(this->assetDirectory.filename().string().c_str())) {
                if (ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                    this->currentDirectory = this->assetDirectory;
                }
                dfsOverFolders(this->assetDirectory);
            }

            ImGui::NextColumn();
            
            ImGui::BeginChild("Inner", { 0, 0 }, false, ImGuiWindowFlags_NoScrollbar);
                static float iconScale = 24;

                ImGui::PushStyleColor(ImGuiCol_Button, { 0, 0, 0, 0 });
                if (this->currentDirectory.string() != this->assetDirectory.string()) {
                    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.38f, 0.38f, 0.38f, 0.50f));
                    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.67f, 0.67f, 0.67f, 0.39f));
                }
                else {
                    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, { 0, 0, 0, 0 });
                    ImGui::PushStyleColor(ImGuiCol_ButtonActive, { 0, 0, 0, 0 });
                }

                ImGui::ImageButtonEx(
                                            100,
                                            (ImTextureID)Odysseus::Texture2D::loadTextureFromFile(
                                                (Folder::getFolderPath("Icons").string() + "/leftArrow.png").c_str()
                                            ).ID, 
                                            { iconScale, iconScale },
                                            { 0, 0 },
                                            { 1, 1 },
                                            { 0, 0 },
                                            { 0, 0, 0, 0 },
                                            { 1, 1, 1, 1 }
                                    );
                
                if (this->currentDirectory.string() != this->assetDirectory.string())
                    if (ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                        currentDirectory = currentDirectory.parent_path();

                        actionIndex -= 1;
                    }

                ImGui::SameLine();

                ImGui::ImageButtonEx(
                                            100,
                                            (ImTextureID)Odysseus::Texture2D::loadTextureFromFile(
                                                (Folder::getFolderPath("Icons").string() + "/leftArrow.png").c_str()
                                            ).ID, 
                                            { iconScale, iconScale },
                                            { 1, 1 },
                                            { 0, 0 },
                                            { 0, 0 },
                                            { 0, 0, 0, 0 },
                                            { 1, 1, 1, 1 }
                                    );

                if (actions.size() > 1 && (actionIndex + 1) < actions.size())
                    if (ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left))
                        currentDirectory = actions[++actionIndex];

                ImGui::SameLine();

                ImGui::ImageButtonEx(
                                            100,
                                            (ImTextureID)Odysseus::Texture2D::loadTextureFromFile(
                                                (Folder::getFolderPath("Icons").string() + "/rotate.png").c_str()
                                            ).ID, 
                                            { iconScale, iconScale },
                                            { 1, 1 },
                                            { 0, 0 },
                                            { 0, 0 },
                                            { 0, 0, 0, 0 },
                                            { 1, 1, 1, 1 }
                                    );
                
                if (ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left))
                    std::fill_n(filter.InputBuf, 256, 0);

                ImGui::PopStyleColor(3);

                ImGui::SameLine();

                // TODO: Implement filter of folders and files
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.4f, 0.4f, 0.4f, 1.00f));
                float filterWidth = ImGui::GetContentRegionAvail().x / 4 + 40;
                filter.Draw(" ", filterWidth);
                static std::string filterContent(filter.InputBuf);
                filterContent = filter.InputBuf;
                std::for_each(filterContent.begin(), filterContent.end(), [](char & c) {
                    c = ::tolower(c);
                });
                ImGui::PopStyleColor();

                ImGui::SameLine();
                
                ImGui::Text("Assets");

                ImGui::Separator();

                // TODO: move this settings inside a file in order to let the user set his custom values
                static float padding = 22;
                static float thumbnailSize = ImGui::GetContentRegionAvail().x / 9;

                float cellSize = thumbnailSize + padding;
                float panelWidth = ImGui::GetContentRegionAvail().x;

                int columnCount = (int)(panelWidth / cellSize) < 1 ? 1 : (int)(panelWidth / cellSize);
                ImGui::Columns(columnCount, 0, false);

                auto index = 0;

                for (auto& directory : std::filesystem::directory_iterator(currentDirectory)) {
                    auto& path = directory.path();
                    auto relativePath = std::filesystem::relative(path, currentDirectory);
                    std::string filenameString = relativePath.filename().string();
                    std::string lowercaseFilenameString(filenameString);
                    std::for_each(lowercaseFilenameString.begin(), lowercaseFilenameString.end(), [](char & c) {
                        c = ::tolower(c);
                    });

                    ImGui::PushStyleColor(ImGuiCol_Button, {0, 0, 0, 0});
                    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.38f, 0.38f, 0.38f, 0.50f));
                    if(directory.is_directory() && (lowercaseFilenameString.find(filterContent) != std::string::npos)) {
                        ImGui::ImageButtonEx(
                                                ++index,
                                                (ImTextureID)Odysseus::Texture2D::loadTextureFromFile(
                                                    (Folder::getFolderPath("Icons").string() + "/folder.png").c_str()
                                                ).ID, 
                                                { thumbnailSize, thumbnailSize },
                                                { 1, 1 },
                                                { 0, 0 },
                                                { 10, 10 },
                                                { 0, 0, 0, 0 },
                                                { 1, 1, 1, 1 }
                                            );
                        if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left))
                        {
                            currentDirectory = path;
                         
                            if (actionIndex == 0 && actions.size() > 1) {
                                actions.clear();
                                actions.push_back(this->assetDirectory);
                            }

                            actions.push_back(path);
                            actionIndex += 1;
                        }
                    }
                    else if (lowercaseFilenameString.find(filterContent) != std::string::npos){
                        ImGui::ImageButtonEx(
                                                    ++index,
                                                    (ImTextureID)Odysseus::Texture2D::loadTextureFromFile(
                                                        (Folder::getFolderPath("Icons").string() + "/document.png").c_str()
                                                    ).ID, 
                                                    { thumbnailSize, thumbnailSize },
                                                    { 1, 1 },
                                                    { 0, 0 },
                                                    { 10, 10 },
                                                    { 0, 0, 0, 0 },
                                                    { 1, 1, 1, 1 }
                                            );
                        if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left))
                            system(path.string().c_str());
                    }
                    ImGui::PopStyleColor(2);

                    if (lowercaseFilenameString.find(filterContent) != std::string::npos) {
                        ImGui::TextWrapped(filenameString.c_str());
                        ImGui::NextColumn();
                    }

                }

                ImGui::Columns(1);
            ImGui::EndChild();

            ImGui::Columns(1);
        ImGui::End();
    }

    void Window::dfsOverFolders(std::filesystem::path sourceFolder, int index)
    {
        for (auto& directory : std::filesystem::directory_iterator(sourceFolder)) {
            float marginX = 15.0f * static_cast<float>(index);
            if (directory.is_directory()) {
                ImGui::Dummy({ marginX, 0 });
                ImGui::SameLine();
                
                if (countNestedFolders(directory.path()) > 0) {
                    bool isOpen = ImGui::CollapsingHeader(directory.path().filename().string().c_str());
                    if (!isOpen && ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                            this->currentDirectory = directory.path();
                    }
                    if(isOpen) {
                        dfsOverFolders(directory.path(), index + 1);
                    }
                }
                else {
                    ImGui::CollapsingHeader(directory.path().filename().string().c_str(), ImGuiTreeNodeFlags_Bullet);
                    if (ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                        this->currentDirectory = directory.path();
                    }
                }
            }
        }
    }

    int Window::countNestedFolders(std::filesystem::path sourceFolder)
    {
        auto counter = 0;

        for (auto& directory : std::filesystem::directory_iterator(sourceFolder)) {
            if (directory.is_directory())
                ++counter;
        }

        return counter;
    }

    Window::~Window()
    {
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();

        glfwTerminate();
    }

}