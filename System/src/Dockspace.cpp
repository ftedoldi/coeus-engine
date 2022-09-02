#include "../Dockspace.hpp"

#include <EditorCamera.hpp>
#include <Transform.hpp>

#include <Model.hpp>

#include <SerializableClass.hpp>

#include <Serializer/Serializer.hpp>

#include <Utils/WindowUtils.hpp>
#include <Utils/GUI.hpp>

#include <regex>

// TODO: Refactor all of this
// TODO: Implement file cancellation within the editor
namespace System {

    Dockspace::Dockspace()
    {
        this->transformToShow = nullptr;

        this->console = new EditorLayer::Console();
        Debug::mainConsole = this->console;

        statusBar = new EditorLayer::StatusBar();
        mainMenuBar = new EditorLayer::MainMenuBar(this->statusBar);

        Folder::assetDirectory = Folder::getFolderPath("Assets");
        Folder::currentDirectory = Folder::assetDirectory;

        initializeButtonImageTextures();

        gizmoOperation = ImGuizmo::OPERATION::TRANSLATE;
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
        buttonImages.modelTextureID = Odysseus::Texture2D::loadTextureFromFile(
                                                                                    (Folder::getFolderPath("Icons").string() + "/model.png").c_str(), 
                                                                                    true
                                                                                ).ID;
    }

    void Dockspace::createDockspace()
    {
        if (Odysseus::SceneManager::activeScene->isRuntimeScene)
            Utils::GUI::setEngineRuntimeStyle();
        else
            Utils::GUI::setEngineEditorStyle();

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

        mainMenuBar->draw();
        createToolMenuBar();
        statusBar->draw();

        ImGui::End();

        handleMousePicking();

        createHierarchyWindow();
        createConsoleWindow();
        createContentBrowser();
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
                        statusBar->addStatus("Translate operation selected", EditorLayer::StatusBarTextColor::GREEN);
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
                        statusBar->addStatus("Scale operation selected", EditorLayer::StatusBarTextColor::GREEN);
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
                        statusBar->addStatus("Rotate operation selected", EditorLayer::StatusBarTextColor::GREEN);
                    }

                    ImGui::SameLine();
                    cursor = ImGui::GetCursorPos();
                    ImGui::SetCursorPos({ ImGui::GetContentRegionAvail().x / 2, cursor.y });

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

                    if (
                            ((ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) ||
                            Input::keyboard->getPressedKey() == GLFW_KEY_P) &&
                            !Odysseus::SceneManager::activeScene->isRuntimeScene
                        ) {
                        statusBar->addStatus("Starting simulation...");
                        Odysseus::Scene* simulationScene = new Odysseus::Scene(Odysseus::SceneManager::activeScene, Odysseus::SceneState::RUNNING, true);
                        Odysseus::SceneManager::activeScene = simulationScene;
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

                    if (    
                            (ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) && 
                            Odysseus::SceneManager::activeScene->isRuntimeScene &&
                            Odysseus::SceneManager::activeScene->status == Odysseus::SceneState::RUNNING
                        ) {
                        statusBar->addStatus("Pausing simulation...");
                        Odysseus::SceneManager::activeScene->status = Odysseus::SceneState::PAUSED;
                    } 
                    else if (
                                (ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) && 
                                Odysseus::SceneManager::activeScene->isRuntimeScene &&
                                Odysseus::SceneManager::activeScene->status == Odysseus::SceneState::PAUSED
                            )
                    {
                        statusBar->addStatus("Resuming simulation...");
                        Odysseus::SceneManager::activeScene->status = Odysseus::SceneState::RUNNING;
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

                    if (
                            (ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) && 
                            Odysseus::SceneManager::activeScene->isRuntimeScene &&
                            Odysseus::SceneManager::activeScene->status != Odysseus::SceneState::EDITOR
                        ) {
                        statusBar->addStatus("Stopping simulation...");
                        auto serializedScene = Odysseus::SceneManager::activeScene;
                        Odysseus::SceneManager::activeScene->status = Odysseus::SceneState::STOPPED;
                        System::Serialize::Serializer serializer = System::Serialize::Serializer();
                        serializer.deserialize(Odysseus::SceneManager::_loadedScenes[Odysseus::SceneManager::_loadedScenes.size() - 1]->path);
                        Odysseus::SceneManager::_loadedScenes.pop_back();
                        delete serializedScene;
                    }

                    ImGui::PopStyleColor(3);
                ImGui::EndMenuBar();
            }
            ImGui::End();
        }

        ImGui::PopStyleColor();
        ImGui::PopStyleVar();
    }

    void Dockspace::createHierarchyWindow()
    {
        static Odysseus::Transform* selectedItem = nullptr;

        ImGui::Begin("Hierarchy");

            // TODO: Draggable Scene Items
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

                if (Odysseus::Transform::CountNestedChildren(Odysseus::SceneManager::activeScene->objectsInScene[i]->transform)) {
                    auto isOpen = ImGui::TreeNodeEx(
                                                        std::to_string(Odysseus::SceneManager::activeScene->objectsInScene[i]->ID).c_str(), 
                                                        ImGuiTreeNodeFlags_CollapsingHeader, 
                                                        Odysseus::SceneManager::activeScene->objectsInScene[i]->transform->name.c_str()
                                                    );
                    
                    if (ImGui::IsItemHovered() && ImGui::IsItemClicked(ImGuiMouseButton_Left)) {
                        this->transformToShow = Odysseus::SceneManager::activeScene->objectsInScene[i]->transform;
                        Utils::GUI::loadInspectorParameters(this->transformToShow);
                    }
                    
                    if (isOpen)
                        Utils::GUI::displayChildrenOfTransform(Odysseus::SceneManager::activeScene->objectsInScene[i]->transform, this->transformToShow);

                } else {
                    ImGui::TreeNodeEx(
                                        std::to_string(Odysseus::SceneManager::activeScene->objectsInScene[i]->ID).c_str(), 
                                        ImGuiTreeNodeFlags_CollapsingHeader | ImGuiTreeNodeFlags_Bullet, 
                                        Odysseus::SceneManager::activeScene->objectsInScene[i]->transform->name.c_str()
                                    );
                    if (ImGui::IsItemHovered() && ImGui::IsItemClicked()) {
                        this->transformToShow = Odysseus::SceneManager::activeScene->objectsInScene[i]->transform;
                        Utils::GUI::loadInspectorParameters(this->transformToShow);
                    }
                }

                if (ImGui::IsItemHovered() && ImGui::IsItemClicked(ImGuiMouseButton_Right))
                    selectedItem = Odysseus::SceneManager::activeScene->objectsInScene[i]->transform;
            }
        ImGui::End();
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
                static char transformName[256] =  { 0 };
                if (transformToShow != lastTransform)
                    strcpy(transformName, this->transformToShow->name.c_str());

                ImGui::InputText("Name", transformName, sizeof(char) * 256, ImGuiInputTextFlags_AutoSelectAll);

                if (ImGui::IsItemActive())
                    this->transformToShow->name = transformName;

                if (glfwGetKey(Window::window, GLFW_KEY_ENTER))
                    ImGui::SetWindowFocus(NULL);
                    
                ImGui::Separator();
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
                for (int i = 0; i < Utils::GUI::inspectorParameters.size(); i++) {
                    #pragma warning(push)
                    #pragma warning(disable : 4312)
                    if (Utils::GUI::inspectorParameters[i]->hasEditorTexture())
                    {
                        ImGui::Image((ImTextureID)Utils::GUI::inspectorParameters[i]->getEditorTextureID(), {12, 12});
                        ImGui::SameLine();
                    }
                    #pragma warning(pop)

                    ImGui::Text(Utils::GUI::inspectorParameters[i]->toString().c_str());
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
                    Utils::GUI::inspectorParameters[i]->showComponentFieldsInEditor();
                    ImGui::PopStyleColor(3);
                    ImGui::Separator();

                    if (shouldDeleteComponent)
                    {
                        std::cout << "Deleting: " << Utils::GUI::inspectorParameters[i]->toString() << std::endl;
                        this->transformToShow->sceneObject->removeComponentWithIndex(i);
                        Utils::GUI::loadInspectorParameters(this->transformToShow);
                    }
                }

                if (Utils::GUI::inspectorParameters.size() > this->transformToShow->sceneObject->_container->components.size())
                    Utils::GUI::loadInspectorParameters(this->transformToShow);

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
                                tmp->start();
                                Utils::GUI::inspectorParameters.push_back(tmp);
                            }
                            else
                            {
                                delete newComponent;
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

                    Utils::GUI::inspectorParameters.clear();
                    this->transformToShow = nullptr;

                    serializer.deserialize(pathToLoad);
                }
                else if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("MODEL_FILE"))
                {
                    const char* pathToLoad = static_cast<const char*>(payload->Data);
                    Odysseus::Shader* modelShader = new Odysseus::Shader(".\\Shader\\phongShader.vert", ".\\Shader\\phongShader.frag");
                    Odysseus::Model myModel(pathToLoad, modelShader, false);
                    Odysseus::SceneManager::initializeActiveScene();
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
                statusBar->addStatus("Could not decompose transformation matrix, please try again!", EditorLayer::StatusBarTextColor::RED);
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
                ImGui::Image((ImTextureID)Window::sceneFrameBuffer->texturesID[0], size, ImVec2(0, 1), ImVec2(1, 0));
                #pragma warning(pop)

                dList->AddCallback(ImDrawCallback_ResetRenderState, nullptr);
            ImGui::EndChild();
        ImGui::End();
    }
    
    void Dockspace::createProjectSettingsWindow()
    {
        // TODO: Implement me
        ImGui::Begin("Project Settings");
            static char pathOfStandardScene[256] = { "" };
            strcpy(pathOfStandardScene, Odysseus::SceneManager::activeScene->path.c_str());
            ImGui::InputText("Default Scene Path", pathOfStandardScene, sizeof(char) * 256);

            char sceneName[256] = { 0 };
            strcpy(sceneName, Odysseus::SceneManager::activeScene->name.c_str());
            ImGui::InputText("Scene Name", sceneName, sizeof(char) * 256);

            if (ImGui::IsItemActive())
                Odysseus::SceneManager::activeScene->name = sceneName;

        ImGui::End();
    }

    void Dockspace::createContentBrowser()
    {
        static int actionIndex = 0;
        static std::vector<std::filesystem::path> actions = { Folder::assetDirectory };

        static ImGuiTextFilter filter;
        
        ImGui::Begin("Project", NULL, ImGuiWindowFlags_NoScrollbar);
            ImGui::Columns(2);

            static bool isFirstOpening = true;
            if (isFirstOpening) 
            {
                ImGui::SetColumnWidth(0, ImGui::GetContentRegionAvail().x / 1.8);
                isFirstOpening = false;
            }

            if(ImGui::CollapsingHeader(Folder::assetDirectory.filename().string().c_str())) {
                if (ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                    Folder::currentDirectory = Folder::assetDirectory;
                }
                Utils::GUI::displayFoldersAtPath(Folder::assetDirectory, Folder::currentDirectory);
            }

            ImGui::NextColumn();
            
            ImGui::BeginChild("Inner", { 0, 0 }, false, ImGuiWindowFlags_NoScrollbar);
                static float iconScale = 24;

                ImGui::PushStyleColor(ImGuiCol_Button, { 0, 0, 0, 0 });
                if (Folder::currentDirectory.string() != Folder::assetDirectory.string()) {
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
                
                if (Folder::currentDirectory.string() != Folder::assetDirectory.string())
                    if (ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                        Folder::currentDirectory = Folder::currentDirectory.parent_path();

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
                        Folder::currentDirectory = actions[++actionIndex];

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

                for (auto& directory : std::filesystem::directory_iterator(Folder::currentDirectory)) {
                    auto& path = directory.path();
                    auto relativePath = std::filesystem::relative(path, Folder::currentDirectory);
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
                            Folder::currentDirectory = path;
                         
                            if (actionIndex == 0 && actions.size() > 1) {
                                actions.clear();
                                actions.push_back(Folder::assetDirectory);
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

                                Utils::GUI::inspectorParameters.clear();
                                this->transformToShow = nullptr;

                                serializer.deserialize(path.string());
                            }

                            static std::string currentPath("");
                            if (ImGui::IsItemHovered())
                                currentPath = path.string();

                            if (ImGui::BeginDragDropSource())
                            {
                                ImGui::SetDragDropPayload("SCENE_FILE", currentPath.c_str(), (strlen(currentPath.c_str()) + 1) * sizeof(char));
                                ImGui::EndDragDropSource();
                            }
                            else
                                currentPath = "";

                        }
                        else if (lowercaseFilenameString.find(".obj") != std::string::npos || lowercaseFilenameString.find(".fbx") != std::string::npos)
                        {
                            #pragma warning(push)
                            #pragma warning(disable : 4312)                
                            ImGui::ImageButton(
                                                    (ImTextureID)buttonImages.modelTextureID, 
                                                    { thumbnailSize, thumbnailSize },
                                                    { 0, 0 },
                                                    { 1, 1 },
                                                    10,
                                                    { 0, 0, 0, 0 },
                                                    { 1, 1, 1, 1 }
                                                );
                            #pragma warning(pop)

                            static std::string currentPath("");
                            if (ImGui::IsItemHovered())
                                currentPath = path.string();
                            
                            if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left))
                            {
                                Odysseus::Shader* modelShader = new Odysseus::Shader(".\\Shader\\phongShader.vert", ".\\Shader\\phongShader.frag");
                                Odysseus::Model myModel(path.string(), modelShader, false);
                                Odysseus::SceneManager::initializeActiveScene();
                            }

                            if (ImGui::BeginDragDropSource())
                            {
                                ImGui::SetDragDropPayload("MODEL_FILE", currentPath.c_str(), (strlen(currentPath.c_str()) + 1) * sizeof(char));
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
}