#include "../Dockspace.hpp"

#include <EditorCamera.hpp>
#include <Transform.hpp>

#include <SerializableClass.hpp>

#include <Serializer/Serializer.hpp>

namespace System {

    Dockspace::Dockspace()
    {
        this->transformToShow = nullptr;

        this->console = new Console();
        Debug::mainConsole = this->console;

        this->assetDirectory = Folder::getFolderPath("Assets");
        this->currentDirectory = this->assetDirectory;

        initializeButtonImageTextures();

        gizmoOperation = ImGuizmo::OPERATION::TRANSLATE;

        statusBar = new StatusBar();
    }

    void Dockspace::initializeButtonImageTextures() 
    {
        buttonImages.translateTextureID = Odysseus::Texture2D::loadTextureFromFile(
                                                                                    (Folder::getFolderPath("Icons").string() + "/translate.png").c_str(), 
                                                                                    true
                                                                                ).ID;
        buttonImages.rotateTextureID = Odysseus::Texture2D::loadTextureFromFile(
                                                                                    (Folder::getFolderPath("Icons").string() + "/rotation.png").c_str(), 
                                                                                    true
                                                                                ).ID;
        buttonImages.scaleTextureID = Odysseus::Texture2D::loadTextureFromFile(
                                                                                    (Folder::getFolderPath("Icons").string() + "/resize.png").c_str(), 
                                                                                    true
                                                                                ).ID;
        buttonImages.playTextureID = Odysseus::Texture2D::loadTextureFromFile(
                                                                                    (Folder::getFolderPath("Icons").string() + "/play.png").c_str(), 
                                                                                    true
                                                                                ).ID;
        buttonImages.pauseTextureID = Odysseus::Texture2D::loadTextureFromFile(
                                                                                    (Folder::getFolderPath("Icons").string() + "/pause.png").c_str(), 
                                                                                    true
                                                                                ).ID;
        buttonImages.stopTextureID = Odysseus::Texture2D::loadTextureFromFile(
                                                                                    (Folder::getFolderPath("Icons").string() + "/stop.png").c_str(), 
                                                                                    true
                                                                                ).ID;
        buttonImages.leftArrowTextureID = Odysseus::Texture2D::loadTextureFromFile(
                                                                                    (Folder::getFolderPath("Icons").string() + "/leftArrow.png").c_str(), 
                                                                                    true
                                                                                ).ID;
        buttonImages.reloadTextureID = Odysseus::Texture2D::loadTextureFromFile(
                                                                                    (Folder::getFolderPath("Icons").string() + "/rotate.png").c_str(), 
                                                                                    true
                                                                                ).ID;
        buttonImages.folderTextureID = Odysseus::Texture2D::loadTextureFromFile(
                                                                                    (Folder::getFolderPath("Icons").string() + "/folder.png").c_str(), 
                                                                                    true
                                                                                ).ID;
        buttonImages.documentTextureID = Odysseus::Texture2D::loadTextureFromFile(
                                                                                    (Folder::getFolderPath("Icons").string() + "/document.png").c_str(), 
                                                                                    true
                                                                                ).ID;
        buttonImages.pointLightTextureID = Odysseus::Texture2D::loadTextureFromFile(
                                                                                    (Folder::getFolderPath("Icons").string() + "/pointLight.png").c_str(), 
                                                                                    true
                                                                                ).ID;
        buttonImages.spotLightTextureID = Odysseus::Texture2D::loadTextureFromFile(
                                                                                    (Folder::getFolderPath("Icons").string() + "/spotLight.png").c_str(), 
                                                                                    true
                                                                                ).ID;
        buttonImages.directionalLightTextureID = Odysseus::Texture2D::loadTextureFromFile(
                                                                                    (Folder::getFolderPath("Icons").string() + "/directionalLight.png").c_str(), 
                                                                                    true
                                                                                ).ID;
        buttonImages.areaLightTextureID = Odysseus::Texture2D::loadTextureFromFile(
                                                                                    (Folder::getFolderPath("Icons").string() + "/areaLight.png").c_str(), 
                                                                                    true
                                                                                ).ID;
        buttonImages.removeComponentTextureID = Odysseus::Texture2D::loadTextureFromFile(
                                                                                    (Folder::getFolderPath("Icons").string() + "/remove.png").c_str(), 
                                                                                    true
                                                                                ).ID;
        buttonImages.sceneTextureID = Odysseus::Texture2D::loadTextureFromFile(
                                                                                    (Folder::getFolderPath("Icons").string() + "/scene.png").c_str(), 
                                                                                    true
                                                                                ).ID;
    }

    void Dockspace::createStyle()
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
    }

    void Dockspace::createDockspace()
    {
        createStyle();

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
        ImGuiID dockspace_id = ImGui::GetID("MyDockSpace");
        ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), dockspace_flags);

        static auto first_time = true;
	    if (first_time)
	    {
	    	first_time = false;

	    	ImGui::DockBuilderRemoveNode(dockspace_id); // clear any previous layout
	    	ImGui::DockBuilderAddNode(dockspace_id, dockspace_flags | ImGuiDockNodeFlags_DockSpace);
	    	ImGui::DockBuilderSetNodeSize(dockspace_id, ImVec2(Window::screen.width, Window::screen.height));

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

        createMainMenuBar();
        createToolMenuBar();
        createStatusMenuBar();

        ImGui::End();

        handleMousePicking();

        createHierarchyWindow();
        createConsoleWindow();
        createContentBrowser(); // TODO: find texture icons and add these icons over textures
        createInspectorWindow();
        createSceneWindow();
        createProjectSettingsWindow();
        createGameWindow();
    }

    void Dockspace::handleMousePicking()
    {
        // TODO: Refactor this
        if (ImGui::IsMouseClicked(ImGuiMouseButton_Left) && !ImGuizmo::IsOver() && !ImGuizmo::IsUsing())
        {
            glBindFramebuffer(GL_READ_FRAMEBUFFER, Window::sceneFrameBuffer->ID);
            glReadBuffer(GL_COLOR_ATTACHMENT1);
            float pixelColor[4];
            glReadPixels(Input::mouse.xPositionRelativeToSceneWindow, Input::mouse.yPositionRelativeToSceneWindow, 1, 1, GL_RGBA, GL_FLOAT, &pixelColor);
            if (System::Picking::PickableObject::getPickableObject(pixelColor[0], &Input::mouse.selectedObject))
            {
                this->transformToShow = Input::mouse.selectedObject->transform;
                statusBar->addStatus("Selected Object: " + Input::mouse.selectedObject->transform->name);
            }
            else
                Input::mouse.selectedObject = nullptr;
            glReadBuffer(GL_NONE);
            glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
        }
    }

    // TODO: Setup menu properly
    void Dockspace::createMainMenuBar()
    {
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
                    // ImGui::MenuItem("Padding", NULL, &opt_padding);
                    ImGui::Separator();

                    // if (ImGui::MenuItem("Flag: NoSplit",                "", (dockspace_flags & ImGuiDockNodeFlags_NoSplit) != 0))                 
                    //     dockspace_flags ^= ImGuiDockNodeFlags_NoSplit;

                    // if (ImGui::MenuItem("Flag: NoResize",               "", (dockspace_flags & ImGuiDockNodeFlags_NoResize) != 0))                
                    //     dockspace_flags ^= ImGuiDockNodeFlags_NoResize;

                    // if (ImGui::MenuItem("Flag: NoDockingInCentralNode", "", (dockspace_flags & ImGuiDockNodeFlags_NoDockingInCentralNode) != 0))  
                    //     dockspace_flags ^= ImGuiDockNodeFlags_NoDockingInCentralNode;

                    // if (ImGui::MenuItem("Flag: AutoHideTabBar",         "", (dockspace_flags & ImGuiDockNodeFlags_AutoHideTabBar) != 0))          
                    //     dockspace_flags ^= ImGuiDockNodeFlags_AutoHideTabBar;

                    // if (ImGui::MenuItem("Flag: PassthruCentralNode",    "", (dockspace_flags & ImGuiDockNodeFlags_PassthruCentralNode) != 0, opt_fullscreen)) 
                    //     dockspace_flags ^= ImGuiDockNodeFlags_PassthruCentralNode;

                    // ImGui::Separator();

                    // if (ImGui::MenuItem("Close", NULL, false, p_open != NULL))
                    //     *p_open = false;
                    ImGui::EndMenu();
                }
                ImGui::PopStyleColor();
                ImGui::EndMenuBar();
            }
            
            ImGui::End();
        }
        ImGui::PopStyleColor(3);
    }

    // TODO: At each action add a log for the Status Bar
    void Dockspace::createToolMenuBar()
    {
        ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.43f, 0.43f, 0.50f, 0));
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, {2, 8});
        
        if (ImGui::BeginViewportSideBar("Tool Bar", ImGui::GetMainViewport(), ImGuiDir_Up, 8, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_MenuBar))
        {
            if (ImGui::BeginMenuBar()) {
                    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.38f, 0.38f, 0.38f, 0.50f));
                    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.38f, 0.38f, 0.38f, 0.70f));
                    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.67f, 0.67f, 0.67f, 0.39f));

                    static ImVec2 buttonPadding = ImVec2(4, 4);
                    static ImVec2 buttonSize = ImVec2(18, 18);

                    ImVec2 cursor = ImGui::GetCursorPos();
                    ImGui::SetCursorPos({ cursor.x + 0.5f, cursor.y + 2.5f });

                    #pragma warning(push)
                    #pragma warning(disable : 4312)                
                    ImGui::ImageButtonEx(
                                UUID(),
                                (ImTextureID)buttonImages.translateTextureID, 
                                buttonSize,
                                { 0, 0 },
                                { 1, 1 },
                                buttonPadding,
                                { 0, 0, 0, 0 },
                                { 1, 1, 1, 1 }
                        );
                    #pragma warning(pop)

                    if ((ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) || Input::keyboard->getPressedKey() == GLFW_KEY_T) {
                        gizmoOperation = ImGuizmo::OPERATION::TRANSLATE;
                        statusBar->addStatus("Translate operation selected", TextColor::GREEN);
                    }

                    ImGui::SameLine();
                    cursor = ImGui::GetCursorPos();
                    ImGui::SetCursorPos({ cursor.x, cursor.y });

                    #pragma warning(push)
                    #pragma warning(disable : 4312)                
                    ImGui::ImageButtonEx(
                                UUID(),
                                (ImTextureID)buttonImages.scaleTextureID, 
                                buttonSize,
                                { 0, 0 },
                                { 1, 1 },
                                buttonPadding,
                                { 0, 0, 0, 0 },
                                { 1, 1, 1, 1 }
                        );
                    #pragma warning(pop)

                    if ((ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) || Input::keyboard->getPressedKey() == GLFW_KEY_S) {
                        gizmoOperation = ImGuizmo::OPERATION::SCALE;
                        statusBar->addStatus("Scale operation selected", TextColor::GREEN);
                    }
                    
                    ImGui::SameLine();
                    cursor = ImGui::GetCursorPos();
                    ImGui::SetCursorPos({ cursor.x, cursor.y });

                    #pragma warning(push)
                    #pragma warning(disable : 4312)                
                    ImGui::ImageButtonEx(
                                UUID(),
                                (ImTextureID)buttonImages.rotateTextureID, 
                                buttonSize,
                                { 0, 0 },
                                { 1, 1 },
                                buttonPadding,
                                { 0, 0, 0, 0 },
                                { 1, 1, 1, 1 }
                        );
                    #pragma warning(pop)

                    if ((ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) || Input::keyboard->getPressedKey() == GLFW_KEY_R) {
                        gizmoOperation = ImGuizmo::OPERATION::ROTATE;
                        statusBar->addStatus("Rotate operation selected", TextColor::GREEN);
                    }

                    ImGui::SameLine();
                    cursor = ImGui::GetCursorPos();
                    ImGui::SetCursorPos({ cursor.x + ImGui::GetContentRegionAvail().x / 3, cursor.y });

                    #pragma warning(push)
                    #pragma warning(disable : 4312)                
                    ImGui::ImageButtonEx(
                                UUID(),
                                (ImTextureID)buttonImages.playTextureID, 
                                buttonSize,
                                { 0, 0 },
                                { 1, 1 },
                                buttonPadding,
                                { 0, 0, 0, 0 },
                                { 1, 1, 1, 1 }
                        );
                    #pragma warning(pop)

                    if ((ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) || Input::keyboard->getPressedKey() == GLFW_KEY_P) {
                        statusBar->addStatus("Starting simulation...");
                    }
                    
                    ImGui::SameLine();

                    #pragma warning(push)
                    #pragma warning(disable : 4312)                
                    ImGui::ImageButtonEx(
                                UUID(),
                                (ImTextureID)buttonImages.pauseTextureID, 
                                buttonSize,
                                { 0, 0 },
                                { 1, 1 },
                                buttonPadding,
                                { 0, 0, 0, 0 },
                                { 1, 1, 1, 1 }
                        );
                    #pragma warning(pop)

                    if ((ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left))) {
                        statusBar->addStatus("Pausing simulation...");
                    }
                    
                    ImGui::SameLine();

                    #pragma warning(push)
                    #pragma warning(disable : 4312)                
                    ImGui::ImageButtonEx(
                                UUID(),
                                (ImTextureID)buttonImages.stopTextureID, 
                                buttonSize,
                                { 0, 0 },
                                { 1, 1 },
                                buttonPadding,
                                { 0, 0, 0, 0 },
                                { 1, 1, 1, 1 }
                        );
                    #pragma warning(pop)

                    if ((ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left))) {
                        statusBar->addStatus("Stopping simulation...");
                    }

                    ImGui::PopStyleColor(3);
                ImGui::EndMenuBar();
            }
            ImGui::End();
        }

        ImGui::PopStyleColor();
        ImGui::PopStyleVar();
    }

    void Dockspace::createStatusMenuBar()
    {
        ImGui::PushStyleColor(ImGuiCol_Border, IM_COL32(255,255,255,0));

        if (ImGui::BeginViewportSideBar("Status Bar", ImGui::GetMainViewport(), ImGuiDir_Down, ImGui::GetFrameHeight(), ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_MenuBar))
        {
            if (ImGui::BeginMenuBar()) {
                    static CurrentStatus statusToDisplay = statusBar->errorStatus;

                    if (statusToDisplay.statusTextColor == TextColor::RED)
                        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1, 0, 0, 1));
                    else if (statusToDisplay.statusTextColor == TextColor::GREEN)
                        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0, 1, 0, 1));
                    else if (statusToDisplay.statusTextColor == TextColor::WHITE)
                        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1, 1, 1, 1));
                    else if (statusToDisplay.statusTextColor == TextColor::YELLOW)
                        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1, 1, 0, 1));
                        
                    if (statusBar->getLastStatus().statusText != statusBar->errorStatus.statusText)
                        statusToDisplay = statusBar->popStatus();

                    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + ImGui::GetColumnWidth() - ImGui::CalcTextSize(statusToDisplay.statusText.c_str()).x 
                        - ImGui::GetScrollX() - 2 * ImGui::GetStyle().ItemSpacing.x);

                    ImGui::Text("%s", statusToDisplay.statusText.c_str());
                    ImGui::PopStyleColor();
                ImGui::EndMenuBar();
            }
            ImGui::End();
        }

        ImGui::PopStyleColor();
    }

    void Dockspace::createHierarchyWindow()
    {
        static Odysseus::Transform* selectedItem = nullptr;

        ImGui::Begin("Hierarchy");

            if (ImGui::IsMouseClicked(ImGuiMouseButton_Right))
                ImGui::OpenPopup("PopupMenu");
                
            if (ImGui::BeginPopup("PopupMenu"))
            {
                if(ImGui::Selectable("Add New SceneObject"))
                    Odysseus::SceneObject* o = new Odysseus::SceneObject("New Scene Object");

                if (ImGui::Selectable("Delete SceneObject") && selectedItem != nullptr)
                {
                    Odysseus::SceneManager::activeScene->deleteSceneObject(selectedItem->sceneObject);
                    this->transformToShow = nullptr;
                }

                ImGui::EndPopup();
            }

            for (int i = 0; i < Odysseus::SceneManager::activeScene->objectsInScene.size(); i++) {
                if (
                        Odysseus::SceneManager::activeScene->objectsInScene[i]->transform->parent != nullptr ||
                        !Odysseus::SceneManager::activeScene->objectsInScene[i]->showInEditor
                    )
                    continue;

                if (countNestedChildren(Odysseus::SceneManager::activeScene->objectsInScene[i]->transform)) {
                    auto isOpen = ImGui::TreeNodeEx(
                                                        std::to_string(Odysseus::SceneManager::activeScene->objectsInScene[i]->ID).c_str(), 
                                                        ImGuiTreeNodeFlags_CollapsingHeader, 
                                                        Odysseus::SceneManager::activeScene->objectsInScene[i]->transform->name.c_str()
                                                    );
                    
                    if (ImGui::IsItemHovered() && ImGui::IsItemClicked(ImGuiMouseButton_Left)) {
                        this->transformToShow = Odysseus::SceneManager::activeScene->objectsInScene[i]->transform;
                        this->loadInspectorParameters(this->transformToShow);
                    }
                    
                    if (isOpen) {
                        this->dfsOverChildren(Odysseus::SceneManager::activeScene->objectsInScene[i]->transform);
                    }
                } else {
                    ImGui::TreeNodeEx(
                                        std::to_string(Odysseus::SceneManager::activeScene->objectsInScene[i]->ID).c_str(), 
                                        ImGuiTreeNodeFlags_CollapsingHeader | ImGuiTreeNodeFlags_Bullet, 
                                        Odysseus::SceneManager::activeScene->objectsInScene[i]->transform->name.c_str()
                                    );
                    if (ImGui::IsItemHovered() && ImGui::IsItemClicked()) {
                        this->transformToShow = Odysseus::SceneManager::activeScene->objectsInScene[i]->transform;
                        this->loadInspectorParameters(this->transformToShow);
                    }
                }

                if (ImGui::IsItemHovered() && ImGui::IsItemClicked(ImGuiMouseButton_Right))
                    selectedItem = Odysseus::SceneManager::activeScene->objectsInScene[i]->transform;
            }
        ImGui::End();
    }
    
    void Dockspace::dfsOverChildren(Odysseus::Transform* childrenTransform, int index) 
    {
        for (int j = 0; j < childrenTransform->children.size(); j++) {

            ImGui::Dummy({ 10 * static_cast<float>(index), 0 });
            ImGui::SameLine();

            if (this->countNestedChildren(childrenTransform->children[j])) {
                auto childOpen = ImGui::TreeNodeEx(
                                                    std::to_string(
                                                                    childrenTransform->children[j]->sceneObject->ID
                                                                ).c_str(), 
                                                    ImGuiTreeNodeFlags_CollapsingHeader, 
                                                    childrenTransform->children[j]->name.c_str()
                                                );

                if (ImGui::IsItemHovered() && ImGui::IsItemClicked()) {
                    this->transformToShow = childrenTransform->children[j];
                    this->loadInspectorParameters(this->transformToShow);
                }

                if (childOpen)
                    this->dfsOverChildren(childrenTransform->children[j], index + 1);           
            } else {
                auto childOpen = ImGui::TreeNodeEx(
                                                    std::to_string(
                                                                    childrenTransform->children[j]->sceneObject->ID
                                                                ).c_str(), 
                                                    ImGuiTreeNodeFlags_CollapsingHeader | ImGuiTreeNodeFlags_Bullet, 
                                                    childrenTransform->children[j]->name.c_str()
                                                );

                if (ImGui::IsItemHovered() && ImGui::IsItemClicked()) {
                    this->transformToShow = childrenTransform->children[j];
                    this->loadInspectorParameters(this->transformToShow);
                }
            }
        }
    }

    void Dockspace::loadInspectorParameters(Odysseus::Transform* transformToAnalyze)
    {
        this->inspectorParams.clear();
        for (int k = 0; k < transformToAnalyze->sceneObject->_container->components.size(); k++)
            this->inspectorParams.push_back(transformToAnalyze->sceneObject->_container->components[k]);
    }

    int Dockspace::countNestedChildren(Odysseus::Transform* childrenTransform)
    {
        if (childrenTransform->children.size() == 0)
            return 0;
        
        return childrenTransform->children.size();
    }

    void Dockspace::createConsoleWindow()
    {
        static bool isOpen = true;
        console->Draw("Console", &isOpen);
    }

    void Dockspace::createInspectorWindow()
    {
        static Odysseus::Transform* lastTransform = nullptr;

        ImGui::Begin("Inspector");
            if (this->transformToShow != nullptr) {
                ImGui::Text("Transform");
                float pos[] = { this->transformToShow->position.coordinates.x, this->transformToShow->position.coordinates.y, this->transformToShow->position.coordinates.z };
                ImGui::InputFloat3("Position", pos);
                this->transformToShow->position = Athena::Vector3(pos[0], pos[1], pos[2]);

                Athena::Vector3 rotationAsVector = this->transformToShow->eulerRotation;
                float rotation[3] = { rotationAsVector[0], rotationAsVector[1], rotationAsVector[2]};
                ImGui::InputFloat3("Rotation", rotation);
                this->transformToShow->eulerRotation = Athena::Vector3(rotation[0], rotation[1], rotation[2]);
                if (ImGui::IsItemEdited())
                    this->transformToShow->rotation = Athena::Quaternion::EulerAnglesToQuaternion(this->transformToShow->eulerRotation);

                float scale[] = { this->transformToShow->localScale.coordinates.x, this->transformToShow->localScale.coordinates.y, this->transformToShow->localScale.coordinates.z };
                ImGui::InputFloat3("Scale", scale);
                this->transformToShow->localScale = Athena::Vector3(scale[0], scale[1], scale[2]);

                // TODO: Do this for every component
                // TODO: Show components serializable fields with protocol buffers
                ImGui::Separator();
                for (int i = 0; i < this->inspectorParams.size(); i++) {
                    #pragma warning(push)
                    #pragma warning(disable : 4312)
                    if (this->inspectorParams[i]->hasEditorTexture())
                    {
                        ImGui::Image((ImTextureID)this->inspectorParams[i]->getEditorTextureID(), {12, 12});
                        ImGui::SameLine();
                    }
                    #pragma warning(pop)

                    ImGui::Text(this->inspectorParams[i]->toString().c_str());
                    ImGui::SameLine(ImGui::GetContentRegionAvail().x - 12);
                    ImGui::PushStyleColor(ImGuiCol_Button, {0, 0, 0, 0});
                    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, {0, 0, 0, 0});
                    ImGui::PushStyleColor(ImGuiCol_ButtonActive, {0, 0, 0, 0});
                    #pragma warning(push)
                    #pragma warning(disable : 4312)
                    bool shouldDeleteComponent = ImGui::ImageButtonEx(
                                i + 1,
                                (ImTextureID)buttonImages.removeComponentTextureID, 
                                { 12, 12 },
                                { 0, 0 },
                                { 1, 1 },
                                { 0, 2 },
                                { 0, 0, 0, 0 },
                                { 1, 1, 1, 1 }
                        );
                    #pragma warning(pop)
                    this->inspectorParams[i]->showComponentFieldsInEditor();
                    ImGui::PopStyleColor(3);
                    ImGui::Separator();

                    if (shouldDeleteComponent)
                    {
                        std::cout << "Deleting: " << this->inspectorParams[i]->toString() << std::endl;
                        this->transformToShow->sceneObject->removeComponentWithIndex(i);
                        this->loadInspectorParameters(this->transformToShow);
                    }
                }

                if (inspectorParams.size() > this->transformToShow->sceneObject->_container->components.size())
                    this->loadInspectorParameters(this->transformToShow);

                static bool isAddComponentOpen = false;

                if (lastTransform != transformToShow)
                {
                    isAddComponentOpen = false;
                    lastTransform = transformToShow;
                }

                // TODO: Remove component button
                if (ImGui::Button("Add Component", { ImGui::GetContentRegionAvail().x, 0 }))
                {
                    isAddComponentOpen = true;
                    ImGui::OpenPopup("Component Popup");
                }

                if (isAddComponentOpen)
                {
                    #pragma warning(push)
                    #pragma warning(disable : 4312)
                    if (ImGui::BeginPopup("Component Popup"))
                    {
                        rttr::type componentType = rttr::type::get_by_name("Component");

                        for (auto derived : componentType.get_derived_classes())
                        {
                            rttr::type t = rttr::type::get_by_name(derived.get_name());
                            rttr::variant v = t.create();

                            System::Component* newComponent = v.convert<System::Component*>();

                            std::string derivedID("##title" + derived.get_name().to_string());
                            auto componentSelectable = ImGui::Selectable(derivedID.c_str());
                            if (newComponent->hasEditorTexture())
                            {
                                ImGui::SameLine();
                                ImGui::Image((ImTextureID)newComponent->getEditorTextureID(), {12, 12});
                            }
                            ImGui::SameLine();
                            ImGui::Text(derived.get_name().to_string().c_str());

                            if (componentSelectable)
                            {
                                auto tmp = this->transformToShow->sceneObject->addCopyOfExistingComponent<System::Component>(newComponent);
                                this->inspectorParams.push_back(tmp);
                            }
                        }
                        
                        ImGui::EndPopup();
                    }
                    #pragma warning(pop)
                }
            }
        ImGui::End();
    }
    
    void Dockspace::createSceneWindow()
    {
        ImGui::Begin("Scene");
            ImGui::BeginChild("Game Render", { 0, 0 }, false, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
                ImDrawList* drawList = ImGui::GetWindowDrawList();
                drawList->AddCallback((ImDrawCallback)&Window::sceneFrameBuffer->framebufferShaderCallback, Window::sceneFrameBuffer);
                ImVec2 wSize = ImGui::GetWindowSize();

                ImGuiWindow* w = ImGui::GetCurrentWindow();

                // std::cout << "Mouse Relative Position: ( " << mousePosRelativeToWindow.x << ", " << mousePosRelativeToWindow.y << " )" << std::endl;

                auto initialFrameBufferWidth = Window::sceneFrameBuffer->frameBufferSize.width;
                auto initialFrameBufferHeight = Window::sceneFrameBuffer->frameBufferSize.height;

                if (wSize.x < wSize.y)
                    Window::sceneFrameBuffer->setNewBufferSize(wSize.y, wSize.y);
                else
                    Window::sceneFrameBuffer->setNewBufferSize(wSize.x, wSize.x);

                ImGui::SetScrollY(0);

                auto imageSize = ImVec2((float)Window::sceneFrameBuffer->frameBufferSize.width, (float)Window::sceneFrameBuffer->frameBufferSize.height);
                auto imagePos = ImVec2((ImGui::GetWindowSize().x - imageSize.x) * 0.5f, (ImGui::GetWindowSize().y - imageSize.y) * 0.5f);
                // ImGui::SetCursorPos(imagePos);

                // TODO: Refactor Mouse Over Texture Logic
                //-----------------------------------------------------MOUSE OVER TEXTURE LOGIC------------------------------------------------------------------------//
                auto xMousePos = ImGui::GetMousePos().x < ImGui::GetWindowPos().x ? 0 : Athena::Math::inverseLerp(ImGui::GetWindowPos().x, ImGui::GetWindowPos().x + ImGui::GetWindowSize().x, ImGui::GetMousePos().x) > 1 ? 1 : Athena::Math::inverseLerp(ImGui::GetWindowPos().x, ImGui::GetWindowPos().x + ImGui::GetWindowSize().x, ImGui::GetMousePos().x);
                auto yMousePos = ImGui::GetMousePos().y < ImGui::GetWindowPos().y ? 0 : Athena::Math::inverseLerp(ImGui::GetWindowPos().y, ImGui::GetWindowPos().y + ImGui::GetWindowSize().y, ImGui::GetMousePos().y) > 1 ? 1 : Athena::Math::inverseLerp(ImGui::GetWindowPos().y, ImGui::GetWindowPos().y + ImGui::GetWindowSize().y, ImGui::GetMousePos().y);
                yMousePos = std::abs(yMousePos - 1.0f);
                ImVec2 mousePosRelativeToWindow = ImVec2(xMousePos, yMousePos);
                Input::mouse.xPositionRelativeToSceneWindow = (xMousePos * wSize.x);
                Input::mouse.yPositionRelativeToSceneWindow = (yMousePos * wSize.y) + std::abs(imagePos.y * 2);
                //----------------------------------------------------------------------------------------------------------------------------------------------------//

                #pragma warning(push)
                #pragma warning(disable : 4312)
                ImGui::Image(
                                (ImTextureID)Window::sceneFrameBuffer->texturesID[0], 
                                { 
                                    (float)Window::sceneFrameBuffer->frameBufferSize.width, 
                                    (float)Window::sceneFrameBuffer->frameBufferSize.height 
                                }, 
                                ImVec2(0, 1), 
                                ImVec2(1, 0)
                            );
                #pragma warning(pop)

                drawList->AddCallback(ImDrawCallback_ResetRenderState, nullptr);

                if (
                        Window::sceneFrameBuffer->frameBufferSize.width != initialFrameBufferWidth 
                        || Window::sceneFrameBuffer->frameBufferSize.height != initialFrameBufferHeight
                    )
                    Window::refreshFrameBuffer = true;

                if (this->transformToShow != nullptr)
                    this->createObjectsGUIZMO();

            ImGui::EndChild();

            if (ImGui::BeginDragDropTarget())
            {
                if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("SCENE_FILE"))
                {
                    const char* pathToLoad = static_cast<const char*>(payload->Data);
                    
                    System::Serialize::Serializer serializer = System::Serialize::Serializer();

                    serializer.deserialize(pathToLoad);
                }

                ImGui::EndDragDropTarget();
            }
        ImGui::End();
    }
    
    void Dockspace::createObjectsGUIZMO()
    {
        ImGuizmo::SetOrthographic(false);
        ImGuizmo::SetDrawlist();
        ImGuizmo::AllowAxisFlip(false);
        ImVec2 size = ImGui::GetContentRegionAvail();
        ImVec2 cursorPos = ImGui::GetCursorScreenPos();  
        ImGuizmo::SetRect(
                            ImGui::GetWindowPos().x, ImGui::GetWindowPos().y, 
                            Window::sceneFrameBuffer->frameBufferSize.width, 
                            Window::sceneFrameBuffer->frameBufferSize.height
                        );

        Athena::Matrix4 projection = Odysseus::EditorCamera::perspective(
                                                                    45.0f, 
                                                                    Window::sceneFrameBuffer->frameBufferSize.width / Window::sceneFrameBuffer->frameBufferSize.height, 
                                                                    0.1f, 
                                                                    100.0f
                                                                );
        projection.data[0] = projection.data[0] / (Window::sceneFrameBuffer->frameBufferSize.width / (float)Window::sceneFrameBuffer->frameBufferSize.height);
        projection.data[5] = projection.data[0];

        Athena::Matrix4 view = Odysseus::SceneManager::activeScene->sceneEditor->editorCamera->getViewMatrix();

        auto worldTransform = Odysseus::Transform::GetWorldTransform(this->transformToShow, this->transformToShow);

        // TODO: Make a method in Matrix4 in order to generate a transform matrix from position, scale and rotation
        Athena::Matrix4 translateMatrix(
                                            Athena::Vector4(1, 0, 0, 0),
                                            Athena::Vector4(0, 1, 0, 0),
                                            Athena::Vector4(0, 0, 1, 0),
                                            Athena::Vector4(
                                                                worldTransform->position.coordinates.x, 
                                                                worldTransform->position.coordinates.y, 
                                                                worldTransform->position.coordinates.z,                                             
                                                                1
                                                            )
                                        );

        Athena::Matrix4 scaleMatrix(
                                        Athena::Vector4(worldTransform->localScale.coordinates.x, 0, 0, 0),
                                        Athena::Vector4(0, worldTransform->localScale.coordinates.y, 0, 0),
                                        Athena::Vector4(0, 0, worldTransform->localScale.coordinates.z, 0),
                                        Athena::Vector4(0, 0, 0,                                        1)
                                    );

        Athena::Matrix4 rotationMatrix = worldTransform->rotation.toMatrix4();

        Athena::Matrix4 objTransform = scaleMatrix * rotationMatrix * translateMatrix;

        //--------------------------------------Snapping Function-----------------------------------------//
        // TODO: Set snapValue customizable - Place it in Options
        bool snap = Input::keyboard->isKeyPressed(Key::RIGHT_CONTROL);
        float snapValue = 0.5f; // Snap to 0.5m for translation/scale
        if (gizmoOperation == ImGuizmo::OPERATION::ROTATE)
            snapValue = 45.0f; // Snap to 45.0f degree for rotation

        float snapValues[3] = { snapValue, snapValue, snapValue };

        ImGuizmo::Manipulate(
                                &view.data[0], 
                                &projection.data[0], 
                                gizmoOperation, 
                                ImGuizmo::LOCAL, 
                                &objTransform.data[0], 
                                nullptr, snap ? snapValues : nullptr
                            );

        if (ImGuizmo::IsUsing()) {
            Athena::Vector3 scale, translate;
            Athena::Quaternion rotation;
            if (Athena::Matrix4::DecomposeMatrixInScaleRotateTranslateComponents(objTransform, scale, rotation, translate))
            {
                Athena::Vector3 deltaTranslation, deltaScale;
                Athena::Quaternion deltaRotation(0, 0, 0, 1);

                Odysseus::Transform* parent = this->transformToShow->parent;

                while (parent != nullptr)
                {
                    deltaTranslation += worldTransform->position - this->transformToShow->position;
                    deltaScale += worldTransform->localScale - this->transformToShow->localScale;
                    // This is how to calculate a quaternion delta
                    deltaRotation = deltaRotation * (worldTransform->rotation * this->transformToShow->rotation.inverse());

                    parent = parent->parent;
                }

                this->transformToShow->position = translate - deltaTranslation;
                this->transformToShow->localScale = scale - deltaScale;
                // This is how to add a delta of quaternions
                this->transformToShow->rotation = rotation.conjugated() * deltaRotation.conjugated();
                this->transformToShow->eulerRotation = this->transformToShow->rotation.toEulerAngles();
            }
            else
            {
                Debug::LogError("Could not decompose transformation matrix, please try again!");
                statusBar->addStatus("Could not decompose transformation matrix, please try again!", TextColor::RED);
            }
        }
    }

    // TODO: setup framebuffer for game scene
    void Dockspace::createGameWindow()
    {
        ImGui::Begin("Game");
            ImGui::BeginChild("Game Render");
                ImDrawList* dList = ImGui::GetWindowDrawList();
                dList->AddCallback((ImDrawCallback)&Window::sceneFrameBuffer->framebufferShaderCallback, Window::sceneFrameBuffer);
                ImGuizmo::SetRect(
                            ImGui::GetWindowPos().x, ImGui::GetWindowPos().y, 
                            Window::sceneFrameBuffer->frameBufferSize.width, 
                            Window::sceneFrameBuffer->frameBufferSize.height
                        );
                ImVec2 size = ImGui::GetWindowSize();

                // Window::frameBufferSize.width = size.x;
                // Window::frameBufferSize.height = size.y;
                
                // Window::sceneFrameBuffer->blit();

                #pragma warning(push)
                #pragma warning(disable : 4312)
                ImGui::Image((ImTextureID)Window::sceneFrameBuffer->texturesID[1], size, ImVec2(0, 1), ImVec2(1, 0));
                #pragma warning(pop)

                dList->AddCallback(ImDrawCallback_ResetRenderState, nullptr);
            ImGui::EndChild();
        ImGui::End();
    }
    
    void Dockspace::createProjectSettingsWindow()
    {
        ImGui::Begin("Project Settings");
        ImGui::Text("Hello, right!");
        ImGui::End();
    }

    // TODO: Fix forward button bug
    void Dockspace::createContentBrowser()
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

                #pragma warning(push)
                #pragma warning(disable : 4312)                
                ImGui::ImageButtonEx(
                                            UUID(),
                                            (ImTextureID)buttonImages.leftArrowTextureID, 
                                            { iconScale, iconScale },
                                            { 0, 0 },
                                            { 1, 1 },
                                            { 0, 0 },
                                            { 0, 0, 0, 0 },
                                            { 1, 1, 1, 1 }
                                    );
                #pragma warning(pop)
                
                if (this->currentDirectory.string() != this->assetDirectory.string())
                    if (ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                        currentDirectory = currentDirectory.parent_path();

                        actionIndex -= 1;
                    }

                ImGui::SameLine();

                #pragma warning(push)
                #pragma warning(disable : 4312)
                ImGui::ImageButtonEx(
                                            UUID(),
                                            (ImTextureID)buttonImages.leftArrowTextureID, 
                                            { iconScale, iconScale },
                                            { 1, 1 },
                                            { 0, 0 },
                                            { 0, 0 },
                                            { 0, 0, 0, 0 },
                                            { 1, 1, 1, 1 }
                                    );
                #pragma warning(pop)

                if (actions.size() > 1 && (actionIndex + 1) < actions.size())
                    if (ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left))
                        currentDirectory = actions[++actionIndex];

                ImGui::SameLine();

                #pragma warning(push)
                #pragma warning(disable : 4312)                
                ImGui::ImageButtonEx(
                                            UUID(),
                                            (ImTextureID)buttonImages.reloadTextureID, 
                                            { iconScale, iconScale },
                                            { 1, 1 },
                                            { 0, 0 },
                                            { 0, 0 },
                                            { 0, 0, 0, 0 },
                                            { 1, 1, 1, 1 }
                                    );
                #pragma warning(pop)
                
                if (ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left))
                    std::fill_n(filter.InputBuf, 256, 0);

                ImGui::PopStyleColor(3);

                ImGui::SameLine();

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

                        #pragma warning(push)
                        #pragma warning(disable : 4312)
                        ImGui::ImageButtonEx(
                                                UUID(),
                                                (ImTextureID)buttonImages.folderTextureID, 
                                                { thumbnailSize, thumbnailSize },
                                                { 0, 0 },
                                                { 1, 1 },
                                                { 10, 10 },
                                                { 0, 0, 0, 0 },
                                                { 1, 1, 1, 1 }
                                            );
                        #pragma warning(pop)

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
                        
                        if (lowercaseFilenameString.find(".coeus") != std::string::npos)
                        {
                            #pragma warning(push)
                            #pragma warning(disable : 4312)                
                            ImGui::ImageButton(
                                                        (ImTextureID)buttonImages.sceneTextureID, 
                                                        { thumbnailSize, thumbnailSize },
                                                        { 0, 0 },
                                                        { 1, 1 },
                                                        10,
                                                        { 0, 0, 0, 0 },
                                                        { 1, 1, 1, 1 }
                                                );
                            #pragma warning(pop)

                            ImGuiWindow* window = ImGui::GetCurrentWindow();
                            ImGui::ButtonBehavior(ImRect(ImGui::GetItemRectMin(), ImGui::GetItemRectMax()), ImGui::GetItemID(), NULL, NULL, 
                             ImGuiButtonFlags_MouseButtonMiddle | ImGuiButtonFlags_MouseButtonRight | ImGuiButtonFlags_MouseButtonLeft);

                            if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left))
                            {
                                System::Serialize::Serializer serializer = System::Serialize::Serializer();

                                serializer.deserialize(path.string());
                            }

                            static std::string currentPath("");
                            if (ImGui::IsItemHovered())
                                currentPath = path.string();

                            if (ImGui::BeginDragDropSource())
                            {
                                // std::cout << currentPath.c_str() << std::endl;
                                // std::cout << currentPath.size() + 8 << std::endl;
                                ImGui::SetDragDropPayload("SCENE_FILE", currentPath.c_str(), (strlen(currentPath.c_str()) + 1) * sizeof(char));
                                ImGui::EndDragDropSource();
                            }
                            else
                                currentPath = "";

                        }
                        else
                        {
                            #pragma warning(push)
                            #pragma warning(disable : 4312)                
                            ImGui::ImageButtonEx(
                                                        UUID(),
                                                        (ImTextureID)buttonImages.documentTextureID, 
                                                        { thumbnailSize, thumbnailSize },
                                                        { 0, 0 },
                                                        { 1, 1 },
                                                        { 10, 10 },
                                                        { 0, 0, 0, 0 },
                                                        { 1, 1, 1, 1 }
                                                );
                            #pragma warning(pop)
                            
                            if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left))
                                system(path.string().c_str());
                        }
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

    void Dockspace::dfsOverFolders(std::filesystem::path sourceFolder, int index)
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

    int Dockspace::countNestedFolders(std::filesystem::path sourceFolder)
    {
        auto counter = 0;

        for (auto& directory : std::filesystem::directory_iterator(sourceFolder)) {
            if (directory.is_directory())
                ++counter;
        }

        return counter;
    }
}