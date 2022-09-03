#include "../Dockspace.hpp"

#include <Editor.hpp>

#include <EditorCamera.hpp>
#include <Transform.hpp>
#include <Model.hpp>

#include <HierarchyWindow.hpp>

#include <SerializableClass.hpp>

#include <Serializer/Serializer.hpp>

#include <Utils/WindowUtils.hpp>
#include <Utils/GUI.hpp>

#include <regex>

// TODO: Refactor all of this
// TODO: Implement file cancellation within the editor
namespace System
{
    Dockspace::Dockspace()
    {
        gizmoOperation = ImGuizmo::OPERATION::TRANSLATE;

        statusBar = new EditorLayer::StatusBar();
        mainMenuBar = new EditorLayer::MainMenuBar(this->statusBar);
        toolBar = new EditorLayer::ToolBar(this->statusBar, gizmoOperation);

        hierarchyWindow = new EditorLayer::HierarchyWindow();
        consoleWindow = new EditorLayer::ConsoleWindow();
        contentBrowserWindow = new EditorLayer::ContentBrowserWindow();

        Folder::assetDirectory = Folder::getFolderPath("Assets");
        Folder::currentDirectory = Folder::assetDirectory;

        initializeButtonImageTextures();
    }

    void Dockspace::initializeButtonImageTextures()
    {
        stbi_set_flip_vertically_on_load(false);

        buttonImages.pointLightTextureID = Odysseus::Texture2D::loadTextureFromFile(
                                               (Folder::getFolderPath("Icons").string() + "/pointLight.png").c_str(),
                                               true)
                                               .ID;
        buttonImages.spotLightTextureID = Odysseus::Texture2D::loadTextureFromFile(
                                              (Folder::getFolderPath("Icons").string() + "/spotLight.png").c_str(),
                                              true)
                                              .ID;
        buttonImages.directionalLightTextureID = Odysseus::Texture2D::loadTextureFromFile(
                                                     (Folder::getFolderPath("Icons").string() + "/directionalLight.png").c_str(),
                                                     true)
                                                     .ID;
        buttonImages.areaLightTextureID = Odysseus::Texture2D::loadTextureFromFile(
                                              (Folder::getFolderPath("Icons").string() + "/areaLight.png").c_str(),
                                              true)
                                              .ID;
        buttonImages.removeComponentTextureID = Odysseus::Texture2D::loadTextureFromFile(
                                                    (Folder::getFolderPath("Icons").string() + "/remove.png").c_str(),
                                                    true)
                                                    .ID;
                                          
        stbi_set_flip_vertically_on_load(true);
    }

    void Dockspace::createDockspace()
    {
        if (Odysseus::SceneManager::activeScene->isRuntimeScene)
            Utils::GUI::setEngineRuntimeStyle();
        else
            Utils::GUI::setEngineEditorStyle();

        Utils::GUI::createDockableArea();

        mainMenuBar->draw();
        toolBar->draw();
        statusBar->draw();

        ImGui::End();

        hierarchyWindow->draw();
        consoleWindow->draw();
        contentBrowserWindow->draw();
        createInspectorWindow();
        createSceneWindow();
        createProjectSettingsWindow();
        createGameWindow();
    }

    void Dockspace::createInspectorWindow()
    {
        static Odysseus::Transform *lastTransform = nullptr;

        ImGui::Begin("Inspector");
        if (Odysseus::SceneManager::activeScene->sceneEditor->selectedTransform != nullptr)
        {
            static char transformName[256] = {0};
            if (Odysseus::SceneManager::activeScene->sceneEditor->selectedTransform != lastTransform)
                strcpy(transformName, Odysseus::SceneManager::activeScene->sceneEditor->selectedTransform->name.c_str());

            ImGui::InputText("Name", transformName, sizeof(char) * 256, ImGuiInputTextFlags_AutoSelectAll);

            if (ImGui::IsItemActive())
                Odysseus::SceneManager::activeScene->sceneEditor->selectedTransform->name = transformName;

            if (glfwGetKey(Window::window, GLFW_KEY_ENTER))
                ImGui::SetWindowFocus(NULL);

            ImGui::Separator();
            ImGui::Text("Transform");
            float pos[] = {
                                Odysseus::SceneManager::activeScene->sceneEditor->selectedTransform->position.coordinates.x, 
                                Odysseus::SceneManager::activeScene->sceneEditor->selectedTransform->position.coordinates.y, 
                                Odysseus::SceneManager::activeScene->sceneEditor->selectedTransform->position.coordinates.z
                            };
            ImGui::InputFloat3("Position", pos);
            Odysseus::SceneManager::activeScene->sceneEditor->selectedTransform->position = Athena::Vector3(pos[0], pos[1], pos[2]);

            Athena::Vector3 rotationAsVector = Odysseus::SceneManager::activeScene->sceneEditor->selectedTransform->eulerRotation;
            float rotation[3] = {rotationAsVector[0], rotationAsVector[1], rotationAsVector[2]};
            ImGui::InputFloat3("Rotation", rotation);
            Odysseus::SceneManager::activeScene->sceneEditor->selectedTransform->eulerRotation = Athena::Vector3(rotation[0], rotation[1], rotation[2]);
            if (ImGui::IsItemEdited())
                Odysseus::SceneManager::activeScene->sceneEditor->selectedTransform->rotation = 
                    Athena::Quaternion::EulerAnglesToQuaternion(
                                                                    Odysseus::SceneManager::activeScene->sceneEditor->selectedTransform->eulerRotation
                                                                );

            float scale[] = {
                                Odysseus::SceneManager::activeScene->sceneEditor->selectedTransform->localScale.coordinates.x, 
                                Odysseus::SceneManager::activeScene->sceneEditor->selectedTransform->localScale.coordinates.y, 
                                Odysseus::SceneManager::activeScene->sceneEditor->selectedTransform->localScale.coordinates.z
                            };
            ImGui::InputFloat3("Scale", scale);
            Odysseus::SceneManager::activeScene->sceneEditor->selectedTransform->localScale = Athena::Vector3(scale[0], scale[1], scale[2]);

            // TODO: Do this for every component
            // TODO: Show components serializable fields with protocol buffers
            ImGui::Separator();
            for (int i = 0; i < Utils::GUI::inspectorParameters.size(); i++)
            {
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
                    {12, 12},
                    {0, 0},
                    {1, 1},
                    {0, 2},
                    {0, 0, 0, 0},
                    {1, 1, 1, 1});
#pragma warning(pop)
                Utils::GUI::inspectorParameters[i]->showComponentFieldsInEditor();
                ImGui::PopStyleColor(3);
                ImGui::Separator();

                if (shouldDeleteComponent)
                {
                    std::cout << "Deleting: " << Utils::GUI::inspectorParameters[i]->toString() << std::endl;
                    Odysseus::SceneManager::activeScene->sceneEditor->selectedTransform->sceneObject->removeComponentWithIndex(i);
                    Utils::GUI::loadInspectorParameters(Odysseus::SceneManager::activeScene->sceneEditor->selectedTransform);
                }
            }

            if (Utils::GUI::inspectorParameters.size() > Odysseus::SceneManager::activeScene->sceneEditor->selectedTransform->
                    sceneObject->_container->components.size())
                Utils::GUI::loadInspectorParameters(Odysseus::SceneManager::activeScene->sceneEditor->selectedTransform);

            static bool isAddComponentOpen = false;

            if (lastTransform != Odysseus::SceneManager::activeScene->sceneEditor->selectedTransform)
            {
                isAddComponentOpen = false;
                lastTransform = Odysseus::SceneManager::activeScene->sceneEditor->selectedTransform;
            }

            // TODO: Remove component button
            if (ImGui::Button("Add Component", {ImGui::GetContentRegionAvail().x, 0}))
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

                        System::Component *newComponent = v.convert<System::Component *>();

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
                            auto tmp = Odysseus::SceneManager::activeScene->sceneEditor->
                                        selectedTransform->sceneObject->addCopyOfExistingComponent<System::Component>(newComponent);
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
        ImGui::BeginChild("Game Render", {0, 0}, false, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
        ImDrawList *drawList = ImGui::GetWindowDrawList();
        drawList->AddCallback((ImDrawCallback)&Window::sceneFrameBuffer->framebufferShaderCallback, Window::sceneFrameBuffer);
        ImVec2 wSize = ImGui::GetWindowSize();

        ImGuiWindow *w = ImGui::GetCurrentWindow();

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
        auto xMousePos = ImGui::GetMousePos().x < ImGui::GetWindowPos().x ? 
                            0 : 
                        Athena::Math::inverseLerp(ImGui::GetWindowPos().x, ImGui::GetWindowPos().x + ImGui::GetWindowSize().x, ImGui::GetMousePos().x) > 1 ? 
                            1 : Athena::Math::inverseLerp(ImGui::GetWindowPos().x, ImGui::GetWindowPos().x + ImGui::GetWindowSize().x, ImGui::GetMousePos().x);
        auto yMousePos = ImGui::GetMousePos().y < ImGui::GetWindowPos().y ? 
                            0 : 
                        Athena::Math::inverseLerp(ImGui::GetWindowPos().y, ImGui::GetWindowPos().y + ImGui::GetWindowSize().y, ImGui::GetMousePos().y) > 1 ? 
                            1 : Athena::Math::inverseLerp(ImGui::GetWindowPos().y, ImGui::GetWindowPos().y + ImGui::GetWindowSize().y, ImGui::GetMousePos().y);

        yMousePos = std::abs(yMousePos - 1.0f);
        
        ImVec2 mousePosRelativeToWindow = ImVec2(xMousePos, yMousePos);
        Input::mouse.xPositionRelativeToSceneWindow = (xMousePos * wSize.x);
        Input::mouse.yPositionRelativeToSceneWindow = (yMousePos * wSize.y) + std::abs(imagePos.y * 2);
        //----------------------------------------------------------------------------------------------------------------------------------------------------//

#pragma warning(push)
#pragma warning(disable : 4312)
        ImGui::Image(
            (ImTextureID)Window::sceneFrameBuffer->texturesID[0],
            {(float)Window::sceneFrameBuffer->frameBufferSize.width,
             (float)Window::sceneFrameBuffer->frameBufferSize.height},
            ImVec2(0, 1),
            ImVec2(1, 0));
#pragma warning(pop)

        drawList->AddCallback(ImDrawCallback_ResetRenderState, nullptr);

        if (
                Window::sceneFrameBuffer->frameBufferSize.width != initialFrameBufferWidth ||
                Window::sceneFrameBuffer->frameBufferSize.height != initialFrameBufferHeight
            )
            Window::refreshFrameBuffer = true;

        if (Odysseus::SceneManager::activeScene->sceneEditor->selectedTransform != nullptr)
            this->createObjectsGUIZMO();

        ImGui::EndChild();

        if (ImGui::BeginDragDropTarget())
        {
            if (const ImGuiPayload *payload = ImGui::AcceptDragDropPayload("SCENE_FILE"))
            {
                const char *pathToLoad = static_cast<const char *>(payload->Data);

                System::Serialize::Serializer serializer = System::Serialize::Serializer();

                Utils::GUI::inspectorParameters.clear();
                Odysseus::SceneManager::activeScene->sceneEditor->selectedTransform = nullptr;

                serializer.deserialize(pathToLoad);
            }
            else if (const ImGuiPayload *payload = ImGui::AcceptDragDropPayload("MODEL_FILE"))
            {
                const char *pathToLoad = static_cast<const char *>(payload->Data);
                Odysseus::Shader *modelShader = new Odysseus::Shader(".\\Shader\\phongShader.vert", ".\\Shader\\phongShader.frag");
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
            Window::sceneFrameBuffer->frameBufferSize.height);

        Athena::Matrix4 projection = Odysseus::EditorCamera::perspective(
            45.0f,
            Window::sceneFrameBuffer->frameBufferSize.width / Window::sceneFrameBuffer->frameBufferSize.height,
            0.1f,
            100.0f);
        projection.data[0] = projection.data[0] / (Window::sceneFrameBuffer->frameBufferSize.width / (float)Window::sceneFrameBuffer->frameBufferSize.height);
        projection.data[5] = projection.data[0];

        Athena::Matrix4 view = Odysseus::SceneManager::activeScene->sceneEditor->editorCamera->getViewMatrix();

        auto worldTransform = Odysseus::Transform::GetWorldTransform(
                                                                        Odysseus::SceneManager::activeScene->sceneEditor->selectedTransform, 
                                                                        Odysseus::SceneManager::activeScene->sceneEditor->selectedTransform
                                                                    );

        // TODO: Make a method in Matrix4 in order to generate a transform matrix from position, scale and rotation
        Athena::Matrix4 translateMatrix(
            Athena::Vector4(1, 0, 0, 0),
            Athena::Vector4(0, 1, 0, 0),
            Athena::Vector4(0, 0, 1, 0),
            Athena::Vector4(
                worldTransform->position.coordinates.x,
                worldTransform->position.coordinates.y,
                worldTransform->position.coordinates.z,
                1));

        Athena::Matrix4 scaleMatrix(
            Athena::Vector4(worldTransform->localScale.coordinates.x, 0, 0, 0),
            Athena::Vector4(0, worldTransform->localScale.coordinates.y, 0, 0),
            Athena::Vector4(0, 0, worldTransform->localScale.coordinates.z, 0),
            Athena::Vector4(0, 0, 0, 1));

        Athena::Matrix4 rotationMatrix = worldTransform->rotation.toMatrix4();

        Athena::Matrix4 objTransform = scaleMatrix * rotationMatrix * translateMatrix;

        //--------------------------------------Snapping Function-----------------------------------------//
        // TODO: Set snapValue customizable - Place it in Options
        bool snap = Input::keyboard->isKeyPressed(Key::RIGHT_CONTROL);
        float snapValue = 0.5f; // Snap to 0.5m for translation/scale
        if (gizmoOperation == ImGuizmo::OPERATION::ROTATE)
            snapValue = 45.0f; // Snap to 45.0f degree for rotation

        float snapValues[3] = {snapValue, snapValue, snapValue};

        ImGuizmo::Manipulate(
            &view.data[0],
            &projection.data[0],
            gizmoOperation,
            ImGuizmo::LOCAL,
            &objTransform.data[0],
            nullptr, snap ? snapValues : nullptr);

        if (ImGuizmo::IsUsing())
        {
            Athena::Vector3 scale, translate;
            Athena::Quaternion rotation;
            if (Athena::Matrix4::DecomposeMatrixInScaleRotateTranslateComponents(objTransform, scale, rotation, translate))
            {
                Athena::Vector3 deltaTranslation, deltaScale;
                Athena::Quaternion deltaRotation(0, 0, 0, 1);

                Odysseus::Transform *parent = Odysseus::SceneManager::activeScene->sceneEditor->selectedTransform->parent;

                while (parent != nullptr)
                {
                    deltaTranslation += worldTransform->position - Odysseus::SceneManager::activeScene->sceneEditor->selectedTransform->position;
                    deltaScale += worldTransform->localScale - Odysseus::SceneManager::activeScene->sceneEditor->selectedTransform->localScale;
                    // This is how to calculate a quaternion delta
                    deltaRotation = deltaRotation * (worldTransform->rotation * 
                        Odysseus::SceneManager::activeScene->sceneEditor->selectedTransform->rotation.inverse());

                    parent = parent->parent;
                }

                Odysseus::SceneManager::activeScene->sceneEditor->selectedTransform->position = translate - deltaTranslation;
                Odysseus::SceneManager::activeScene->sceneEditor->selectedTransform->localScale = scale - deltaScale;
                // This is how to add a delta of quaternions
                Odysseus::SceneManager::activeScene->sceneEditor->selectedTransform->rotation = rotation.conjugated() * deltaRotation.conjugated();
                Odysseus::SceneManager::activeScene->sceneEditor->selectedTransform->eulerRotation = Odysseus::SceneManager::activeScene->sceneEditor->
                    selectedTransform->rotation.toEulerAngles();
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
        ImDrawList *dList = ImGui::GetWindowDrawList();
        dList->AddCallback((ImDrawCallback)&Window::sceneFrameBuffer->framebufferShaderCallback, Window::sceneFrameBuffer);
        ImGuizmo::SetRect(
            ImGui::GetWindowPos().x, ImGui::GetWindowPos().y,
            Window::sceneFrameBuffer->frameBufferSize.width,
            Window::sceneFrameBuffer->frameBufferSize.height);
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
        static char pathOfStandardScene[256] = {""};
        strcpy(pathOfStandardScene, Odysseus::SceneManager::activeScene->path.c_str());
        ImGui::InputText("Default Scene Path", pathOfStandardScene, sizeof(char) * 256);

        char sceneName[256] = {0};
        strcpy(sceneName, Odysseus::SceneManager::activeScene->name.c_str());
        ImGui::InputText("Scene Name", sceneName, sizeof(char) * 256);

        if (ImGui::IsItemActive())
            Odysseus::SceneManager::activeScene->name = sceneName;

        ImGui::End();
    }
}