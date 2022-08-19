#include "../Editor.hpp"

#include <SceneObject.hpp>
#include <EditorCamera.hpp>
#include <EditorCameraMovement.hpp>

namespace System
{
    Editor::Editor()
    {
        Odysseus::SceneObject* editorCameraObject = new Odysseus::SceneObject();
        editorCameraObject->transform->name = "EditorCamera";
        editorCameraObject->transform->position = Athena::Vector3(0, 0, 20);
        this->editorCamera = editorCameraObject->addComponent<Odysseus::EditorCamera>();
        auto cameraMovement = editorCameraObject->addComponent<EditorCameraMovement>();
        cameraMovement->editorCamera = editorCamera;
    }
} // namespace System
