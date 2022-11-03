#include "../InspectorSceneWindow.hpp"

#include <Window.hpp>

#include <Model.hpp>

#include <IO/Input.hpp>
#include <Folder.hpp>
#include <GUI.hpp>
#include <Debug.hpp>

#include <Cubemap.hpp>

#include <EditorCamera.hpp>

namespace EditorLayer
{

    InspectorSceneWindow::InspectorSceneWindow(StatusBar*& mainStatusBar, ImGuizmo::OPERATION& gizmoOp) : gizmoOperation(gizmoOp)
    {
        this->_mainStatusBar = mainStatusBar;
    }

    void InspectorSceneWindow::draw()
    {
        ImGui::Begin("Scene");
        ImGui::BeginChild("Game Render", {0, 0}, false, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
        ImDrawList *drawList = ImGui::GetWindowDrawList();
        drawList->AddCallback((ImDrawCallback)&System::Window::sceneFrameBuffer->framebufferShaderCallback, System::Window::sceneFrameBuffer);
        ImVec2 wSize = ImGui::GetWindowSize();

        ImGuiWindow *w = ImGui::GetCurrentWindow();

        auto initialFrameBufferWidth = System::Window::sceneFrameBuffer->frameBufferSize.width;
        auto initialFrameBufferHeight = System::Window::sceneFrameBuffer->frameBufferSize.height;

        if (wSize.x < wSize.y)
            System::Window::sceneFrameBuffer->setNewBufferSize(wSize.y, wSize.y);
        else
            System::Window::sceneFrameBuffer->setNewBufferSize(wSize.x, wSize.x);

        ImGui::SetScrollY(0);

        auto imageSize = ImVec2((float)System::Window::sceneFrameBuffer->frameBufferSize.width, (float)System::Window::sceneFrameBuffer->frameBufferSize.height);
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
        System::Input::mouse.xPositionRelativeToSceneWindow = (xMousePos * wSize.x);
        System::Input::mouse.yPositionRelativeToSceneWindow = (yMousePos * wSize.y) + std::abs(imagePos.y * 2);
        //----------------------------------------------------------------------------------------------------------------------------------------------------//
#pragma warning(push)
#pragma warning(disable : 4312)
        ImGui::Image(
            (ImTextureID)System::Window::sceneFrameBuffer->postProcessedTexture,
            {(float)System::Window::sceneFrameBuffer->frameBufferSize.width,
             (float)System::Window::sceneFrameBuffer->frameBufferSize.height},
            ImVec2(0, 1),
            ImVec2(1, 0));
#pragma warning(pop)

        drawList->AddCallback(ImDrawCallback_ResetRenderState, nullptr);

        if (
                System::Window::sceneFrameBuffer->frameBufferSize.width != initialFrameBufferWidth ||
                System::Window::sceneFrameBuffer->frameBufferSize.height != initialFrameBufferHeight
            )
            System::Window::refreshFrameBuffer = true;

        if (Odysseus::SceneManager::activeScene->sceneEditor->selectedTransform != nullptr)
            this->createObjectsGUIZMO();

        ImGui::EndChild();

        if (ImGui::BeginDragDropTarget())
        {
            if (const ImGuiPayload *payload = ImGui::AcceptDragDropPayload("SCENE_FILE"))
            {
                const char *pathToLoad = static_cast<const char *>(payload->Data);

                System::Serialize::Serializer serializer = System::Serialize::Serializer();

                System::Utils::GUI::inspectorParameters.clear();
                Odysseus::SceneManager::activeScene->sceneEditor->selectedTransform = nullptr;

                serializer.deserialize(pathToLoad);
            }
            else if (const ImGuiPayload *payload = ImGui::AcceptDragDropPayload("MODEL_FILE"))
            {
                const char *pathToLoad = static_cast<const char *>(payload->Data);

                std::string objectType = pathToLoad;
                objectType = objectType.substr(objectType.find_last_of(".") + 1);

                Odysseus::Model myModel(pathToLoad, Odysseus::Cubemap::currentCubemap->PBRTextureShader, true, objectType);
                Odysseus::SceneManager::initializeActiveScene();
            }

            ImGui::EndDragDropTarget();
        }
        ImGui::End();
    }

    void InspectorSceneWindow::createObjectsGUIZMO()
    {
        ImGuizmo::SetOrthographic(false);
        ImGuizmo::SetDrawlist();
        ImGuizmo::AllowAxisFlip(false);
        ImVec2 size = ImGui::GetContentRegionAvail();
        ImVec2 cursorPos = ImGui::GetCursorScreenPos();
        ImGuizmo::SetRect(
            ImGui::GetWindowPos().x, ImGui::GetWindowPos().y,
            System::Window::sceneFrameBuffer->frameBufferSize.width,
            System::Window::sceneFrameBuffer->frameBufferSize.height);

        Athena::Matrix4 projection = Odysseus::EditorCamera::perspective(
            45.0f,
            System::Window::sceneFrameBuffer->frameBufferSize.width / System::Window::sceneFrameBuffer->frameBufferSize.height,
            0.1f,
            100.0f);
        projection.data[0] = projection.data[0] / (System::Window::sceneFrameBuffer->frameBufferSize.width / (float)System::Window::sceneFrameBuffer->frameBufferSize.height);
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
        bool snap = System::Input::keyboard->isKeyPressed(System::Key::RIGHT_CONTROL);
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
                System::Debug::LogError("Could not decompose transformation matrix, please try again!");
                this->_mainStatusBar->addStatus("Could not decompose transformation matrix, please try again!", EditorLayer::StatusBarTextColor::RED);
            }
        }
    }
} // namespace EditorLayer
