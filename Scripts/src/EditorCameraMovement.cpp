#include "../EditorCameraMovement.hpp"

#include <Window.hpp>
#include <EditorCamera.hpp>

EditorCameraMovement::EditorCameraMovement()
{

}

void EditorCameraMovement::start()
{
    this->movementSpeed = 5.0f;

    this->yaw = -90;
    this->pitch = 0;

    this->mouseSensitivity = .2f;

    this->yValue = 0.0f;

    xMousePos = System::Input::mouse.xPosition;
    yMousePos = System::Input::mouse.yPosition;
}

void EditorCameraMovement::update()
{
    float lastYaw = yaw;
    
    if (glfwGetKey(System::Window::window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
    {
        if (glfwGetKey(System::Window::window, GLFW_KEY_W) == GLFW_PRESS)
            editorCamera->transform->position += editorCamera->Front * movementSpeed * System::Time::deltaTime;
        if (glfwGetKey(System::Window::window, GLFW_KEY_S) == GLFW_PRESS)
            editorCamera->transform->position -= editorCamera->Front * movementSpeed * System::Time::deltaTime;
        if (glfwGetKey(System::Window::window, GLFW_KEY_A) == GLFW_PRESS)
            editorCamera->transform->position -= editorCamera->Right * movementSpeed * System::Time::deltaTime;
        if (glfwGetKey(System::Window::window, GLFW_KEY_D) == GLFW_PRESS)
            editorCamera->transform->position += editorCamera->Right * movementSpeed * System::Time::deltaTime;
        if (glfwGetKey(System::Window::window, GLFW_KEY_Q) == GLFW_PRESS)
        {
            editorCamera->transform->position += Athena::Vector3::up() * movementSpeed * System::Time::deltaTime;
            yValue = editorCamera->transform->position.coordinates.y;
        }
        if (glfwGetKey(System::Window::window, GLFW_KEY_E) == GLFW_PRESS)
        {
            editorCamera->transform->position -= Athena::Vector3::up() * movementSpeed * System::Time::deltaTime;
            yValue = editorCamera->transform->position.coordinates.y;
        }
    }

    if (glfwGetKey(System::Window::window, GLFW_KEY_LEFT_ALT) == GLFW_PRESS)
    {
        editorCamera->transform->position.coordinates.y = yValue;
        yaw += (System::Input::mouse.xPosition - xMousePos) * mouseSensitivity;
        pitch += (yMousePos - System::Input::mouse.yPosition) * mouseSensitivity;

        if (pitch > 89.0f)
            pitch = 89.0f;

        if (pitch < -89.0f)
            pitch = -89.0f;
    }

    xMousePos = System::Input::mouse.xPosition;
    yMousePos = System::Input::mouse.yPosition;

    editorCamera->lookAtPitchYawDirection(pitch, yaw);
}

void EditorCameraMovement::setOrderOfExecution(const short& newOrderOfExecution)
{
    _orderOfExecution = newOrderOfExecution;
}

int EditorCameraMovement::getUniqueID()
{
    return 120902;
}

std::string EditorCameraMovement::toString()
{
    return "Editor Camera Movement";
}

EditorCameraMovement::~EditorCameraMovement()
{
    delete editorCamera;
}