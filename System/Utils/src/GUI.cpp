#include "../GUI.hpp"

#include <Component.hpp>

namespace System::Utils
{
    std::vector<Component*> GUI::inspectorParameters;

    void GUI::setEngineEditorStyle()
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

    void GUI::setEngineRuntimeStyle()
    {
        auto ColorFromBytes = [](uint8_t r, uint8_t g, uint8_t b)
        {
            return ImVec4((float)r / 255.0f, (float)g / 255.0f, (float)b / 255.0f, 1.0f);
        };

        auto& style = ImGui::GetStyle();

        style.Colors[ImGuiCol_Text]                  = ImVec4(1.00f, 1.00f, 1.00f, 1.00f);
        style.Colors[ImGuiCol_TextDisabled]          = ImVec4(0.50f, 0.50f, 0.50f, 1.00f);
        style.Colors[ImGuiCol_WindowBg]              = ImVec4(0.13f - 0.08f, 0.14f - 0.08f, 0.15f - 0.08f, 1.00f);
        style.Colors[ImGuiCol_ChildBg]               = ImVec4(0.13f - 0.08f, 0.14f - 0.08f, 0.15f - 0.08f, 1.00f);
        style.Colors[ImGuiCol_PopupBg]               = ImVec4(0.13f - 0.08f, 0.14f - 0.08f, 0.15f - 0.08f, 1.00f);
        style.Colors[ImGuiCol_Border]                = ImVec4(0.43f - 0.08f, 0.43f - 0.08f, 0.50f - 0.08f, 0.50f);
        style.Colors[ImGuiCol_BorderShadow]          = ImVec4(0.00f - 0.08f, 0.00f - 0.08f, 0.00f - 0.08f, 0.00f);
        style.Colors[ImGuiCol_FrameBg]               = ImVec4(0.25f - 0.08f, 0.25f - 0.08f, 0.25f - 0.08f, 1.00f);
        style.Colors[ImGuiCol_FrameBgHovered]        = ImVec4(0.38f - 0.08f, 0.38f - 0.08f, 0.38f - 0.08f, 1.00f);
        style.Colors[ImGuiCol_FrameBgActive]         = ImVec4(0.67f - 0.08f, 0.67f - 0.08f, 0.67f - 0.08f, 0.39f);
        style.Colors[ImGuiCol_TitleBg]               = ImVec4(0.08f - 0.08f, 0.08f - 0.08f, 0.09f - 0.08f, 1.00f);
        style.Colors[ImGuiCol_TitleBgActive]         = ImVec4(0.08f - 0.08f, 0.08f - 0.08f, 0.09f - 0.08f, 1.00f);
        style.Colors[ImGuiCol_TitleBgCollapsed]      = ImVec4(0.00f - 0.08f, 0.00f - 0.08f, 0.00f - 0.08f, 0.51f);
        style.Colors[ImGuiCol_MenuBarBg]             = ImVec4(0.14f - 0.08f, 0.14f - 0.08f, 0.14f - 0.08f, 1.00f);
        style.Colors[ImGuiCol_ScrollbarBg]           = ImVec4(0.02f - 0.08f, 0.02f - 0.08f, 0.02f - 0.08f, 0.53f);
        style.Colors[ImGuiCol_ScrollbarGrab]         = ImVec4(0.31f - 0.08f, 0.31f - 0.08f, 0.31f - 0.08f, 1.00f);
        style.Colors[ImGuiCol_ScrollbarGrabHovered]  = ImVec4(0.41f - 0.08f, 0.41f - 0.08f, 0.41f - 0.08f, 1.00f);
        style.Colors[ImGuiCol_ScrollbarGrabActive]   = ImVec4(0.51f - 0.08f, 0.51f - 0.08f, 0.51f - 0.08f, 1.00f);
        style.Colors[ImGuiCol_CheckMark]             = ImVec4(0.11f - 0.08f, 0.64f - 0.08f, 0.92f - 0.08f, 1.00f);
        style.Colors[ImGuiCol_SliderGrab]            = ImVec4(0.11f - 0.08f, 0.64f - 0.08f, 0.92f - 0.08f, 1.00f);
        style.Colors[ImGuiCol_SliderGrabActive]      = ImVec4(0.08f - 0.08f, 0.50f - 0.08f, 0.72f - 0.08f, 1.00f);
        style.Colors[ImGuiCol_Button]                = ImVec4(0.25f - 0.08f, 0.25f - 0.08f, 0.25f - 0.08f, 1.00f);
        style.Colors[ImGuiCol_ButtonHovered]         = ImVec4(0.38f - 0.08f, 0.38f - 0.08f, 0.38f - 0.08f, 1.00f);
        style.Colors[ImGuiCol_ButtonActive]          = ImVec4(0.67f - 0.08f, 0.67f - 0.08f, 0.67f - 0.08f, 0.39f);
        style.Colors[ImGuiCol_Header]                = ColorFromBytes(121, 170, 247);
        style.Colors[ImGuiCol_HeaderHovered]         = ColorFromBytes(199, 220, 252);
        style.Colors[ImGuiCol_HeaderActive]          = ColorFromBytes(121, 170, 247);
        style.Colors[ImGuiCol_Separator]             = style.Colors[ImGuiCol_Border];
        style.Colors[ImGuiCol_SeparatorHovered]      = ImVec4(0.41f - 0.08f, 0.42f - 0.08f, 0.44f - 0.08f, 1.00f);
        style.Colors[ImGuiCol_SeparatorActive]       = ImVec4(0.26f - 0.08f, 0.59f - 0.08f, 0.98f - 0.08f, 0.95f);
        style.Colors[ImGuiCol_ResizeGrip]            = ImVec4(0.00f - 0.08f, 0.00f - 0.08f, 0.00f - 0.08f, 0.00f);
        style.Colors[ImGuiCol_ResizeGripHovered]     = ImVec4(0.29f - 0.08f, 0.30f - 0.08f, 0.31f - 0.08f, 0.67f);
        style.Colors[ImGuiCol_ResizeGripActive]      = ImVec4(0.26f - 0.08f, 0.59f - 0.08f, 0.98f - 0.08f, 0.95f);
        style.Colors[ImGuiCol_Tab]                   = ImVec4(0.08f - 0.08f, 0.08f - 0.08f, 0.09f - 0.08f, 0.83f);
        style.Colors[ImGuiCol_TabHovered]            = ImVec4(0.33f - 0.08f, 0.34f - 0.08f, 0.36f - 0.08f, 0.83f);
        style.Colors[ImGuiCol_TabActive]             = ImVec4(0.23f - 0.08f, 0.23f - 0.08f, 0.24f - 0.08f, 1.00f);
        style.Colors[ImGuiCol_TabUnfocused]          = ImVec4(0.08f - 0.08f, 0.08f - 0.08f, 0.09f - 0.08f, 1.00f);
        style.Colors[ImGuiCol_TabUnfocusedActive]    = ImVec4(0.13f - 0.08f, 0.14f - 0.08f, 0.15f - 0.08f, 1.00f);
        style.Colors[ImGuiCol_DockingPreview]        = ImVec4(0.26f - 0.08f, 0.59f - 0.08f, 0.98f - 0.08f, 0.70f);
        style.Colors[ImGuiCol_DockingEmptyBg]        = ImVec4(0.20f - 0.08f, 0.20f - 0.08f, 0.20f - 0.08f, 1.00f);
        style.Colors[ImGuiCol_PlotLines]             = ImVec4(0.61f - 0.08f, 0.61f - 0.08f, 0.61f - 0.08f, 1.00f);
        style.Colors[ImGuiCol_PlotLinesHovered]      = ImVec4(1.00f - 0.08f, 0.43f - 0.08f, 0.35f - 0.08f, 1.00f);
        style.Colors[ImGuiCol_PlotHistogram]         = ImVec4(0.90f - 0.08f, 0.70f - 0.08f, 0.00f - 0.08f, 1.00f);
        style.Colors[ImGuiCol_PlotHistogramHovered]  = ImVec4(1.00f - 0.08f, 0.60f - 0.08f, 0.00f - 0.08f, 1.00f);
        style.Colors[ImGuiCol_TextSelectedBg]        = ImVec4(0.26f - 0.08f, 0.59f - 0.08f, 0.98f - 0.08f, 0.35f);
        style.Colors[ImGuiCol_DragDropTarget]        = ImVec4(0.11f - 0.08f, 0.64f - 0.08f, 0.92f - 0.08f, 1.00f);
        style.Colors[ImGuiCol_NavHighlight]          = ImVec4(0.26f - 0.08f, 0.59f - 0.08f, 0.98f - 0.08f, 1.00f);
        style.Colors[ImGuiCol_NavWindowingHighlight] = ImVec4(1.00f - 0.08f, 1.00f - 0.08f, 1.00f - 0.08f, 0.70f);
        style.Colors[ImGuiCol_NavWindowingDimBg]     = ImVec4(0.80f - 0.08f, 0.80f - 0.08f, 0.80f - 0.08f, 0.20f);
        style.Colors[ImGuiCol_ModalWindowDimBg]      = ImVec4(0.80f - 0.08f, 0.80f - 0.08f, 0.80f - 0.08f, 0.35f);
        style.Colors[ImGuiCol_Header]                = ImVec4(0.0f, 0.0f, 0.0f, 0.f);
        style.Colors[ImGuiCol_HeaderHovered]         = ImVec4(0.38f - 0.08f, 0.38f - 0.08f, 0.38f - 0.08f, 1.00f);
        style.Colors[ImGuiCol_HeaderActive]          = ImVec4(0.67f - 0.08f, 0.67f - 0.08f, 0.67f - 0.08f, 0.39f);

        style.WindowRounding    = 2.0f;
        style.ChildRounding     = 2.0f;
        style.FrameRounding     = 2.0f;
        style.GrabRounding      = 2.0f;
        style.PopupRounding     = 2.0f;
        style.ScrollbarRounding = 2.0f;
        style.TabRounding       = 2.0f;
    }

    void GUI::createDockableArea()
    {
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
    }

    void GUI::displayFoldersAtPath(std::filesystem::path sourceFolder, std::filesystem::path& currentDirectory, int index)
    {
        for (auto& directory : std::filesystem::directory_iterator(sourceFolder)) {
            float marginX = 15.0f * static_cast<float>(index);
            if (directory.is_directory()) {
                ImGui::Dummy({ marginX, 0 });
                ImGui::SameLine();
                
                if (Folder::countNestedFolders(directory.path()) > 0) {
                    bool isOpen = ImGui::CollapsingHeader(directory.path().filename().string().c_str());
                    if (!isOpen && ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                            currentDirectory = directory.path();
                    }
                    if(isOpen) {
                        displayFoldersAtPath(directory.path(), currentDirectory, index + 1);
                    }
                }
                else {
                    ImGui::CollapsingHeader(directory.path().filename().string().c_str(), ImGuiTreeNodeFlags_Bullet);
                    if (ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                        currentDirectory = directory.path();
                    }
                }
            }
        }
    }

    void GUI::displayChildrenOfTransform(Odysseus::Transform* childrenTransform, Odysseus::Transform*& transformToShow, int index)
    {
        for (int j = 0; j < childrenTransform->children.size(); j++) {

            ImGui::Dummy({ 10 * static_cast<float>(index), 0 });
            ImGui::SameLine();

            if (Odysseus::Transform::CountNestedChildren(childrenTransform->children[j])) {
                auto childOpen = ImGui::TreeNodeEx(
                                                    std::to_string(
                                                                    childrenTransform->children[j]->sceneObject->ID
                                                                ).c_str(), 
                                                    ImGuiTreeNodeFlags_CollapsingHeader, 
                                                    childrenTransform->children[j]->name.c_str()
                                                );

                if (ImGui::IsItemHovered() && ImGui::IsItemClicked()) {
                    transformToShow = childrenTransform->children[j];
                    loadInspectorParameters(transformToShow);
                }

                if (childOpen)
                    displayChildrenOfTransform(childrenTransform->children[j], transformToShow, index + 1);           
            } else {
                auto childOpen = ImGui::TreeNodeEx(
                                                    std::to_string(
                                                                    childrenTransform->children[j]->sceneObject->ID
                                                                ).c_str(), 
                                                    ImGuiTreeNodeFlags_CollapsingHeader | ImGuiTreeNodeFlags_Bullet, 
                                                    childrenTransform->children[j]->name.c_str()
                                                );

                if (ImGui::IsItemHovered() && ImGui::IsItemClicked()) {
                    transformToShow = childrenTransform->children[j];
                    loadInspectorParameters(transformToShow);
                }
            }
        }
    }

    void GUI::loadInspectorParameters(Odysseus::Transform* transformToAnalyze)
    {
        inspectorParameters.clear();
        for (int k = 0; k < transformToAnalyze->sceneObject->_container->components.size(); k++)
            inspectorParameters.push_back(transformToAnalyze->sceneObject->_container->components[k]);
    }

}