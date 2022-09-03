#include "../Editor.hpp"

#include <Window.hpp>

#include <SceneObject.hpp>
#include <Transform.hpp>

#include <Dockspace.hpp>
#include <LightInfo.hpp>

#include <EditorCamera.hpp>
#include <EditorCameraMovement.hpp>

#include <IO/Input.hpp>

namespace EditorLayer
{
    Editor::Editor()
    {
        this->setupEditorCamera();
        this->selectedTransform = nullptr;

        Odysseus::LightInfo::resetLightInfo();

        dockedSpace = new System::Dockspace();
    }

    void Editor::setupEditorCamera()
    {
        Odysseus::SceneObject* editorCameraObject = new Odysseus::SceneObject();

        editorCameraObject->transform->name = "EditorCamera";
        editorCameraObject->transform->position = Athena::Vector3(0, 0, 20);
        this->editorCamera = editorCameraObject->addComponent<Odysseus::EditorCamera>();
        auto cameraMovement = editorCameraObject->addComponent<EditorCameraMovement>();
        cameraMovement->editorCamera = editorCamera;
        editorCameraObject->showInEditor = false;
    }

    void Editor::handleMousePicking()
    {
        if (ImGui::IsMouseClicked(ImGuiMouseButton_Left) && !ImGuizmo::IsOver() && !ImGuizmo::IsUsing())
        {
            glBindFramebuffer(GL_READ_FRAMEBUFFER, System::Window::sceneFrameBuffer->ID);
            glReadBuffer(GL_COLOR_ATTACHMENT1);
            float pixelColor[4];
            glReadPixels(System::Input::mouse.xPositionRelativeToSceneWindow, System::Input::mouse.yPositionRelativeToSceneWindow, 1, 1, GL_RGBA, GL_FLOAT, &pixelColor);
            if (System::Picking::PickableObject::getPickableObject(pixelColor[0], &System::Input::mouse.selectedObject))
            {
                Odysseus::SceneManager::activeScene->sceneEditor->selectedTransform = System::Input::mouse.selectedObject->transform;
                System::Utils::GUI::loadInspectorParameters(this->selectedTransform);
                dockedSpace->statusBar->addStatus("Selected Object: " + System::Input::mouse.selectedObject->transform->name);
            }
            else
                System::Input::mouse.selectedObject = nullptr;
            glReadBuffer(GL_NONE);
            glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
        }
    }

    void Editor::onEditorUpdate()
    {
        this->handleMousePicking();
        
        dockedSpace->createDockspace();
    }

} // namespace System
