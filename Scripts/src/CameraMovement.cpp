#include "../CameraMovement.hpp"

CameraMovement::CameraMovement()
{

}

void CameraMovement::start()
{
    this->movementSpeed = 2.0f;

    this->yaw = 0;
    this->pitch = 0;

    this->mouseSensitivity = .5f;

    xMousePos = System::Input::mouse.xPosition;
    yMousePos = System::Input::mouse.yPosition;

    camera->Front = Athena::Vector3(std::cos(yaw), 0, -std::sin(yaw));
    camera->Right = camera->Front.cross(Athena::Vector3::up()).normalized();
}

void CameraMovement::update()
{
    float lastYaw = yaw;

    if (glfwGetKey(System::Window::window, GLFW_KEY_W) == GLFW_PRESS)
        camera->transform->position += camera->Front * movementSpeed * System::Time::deltaTime;
    if (glfwGetKey(System::Window::window, GLFW_KEY_S) == GLFW_PRESS)
        camera->transform->position -= camera->Front * movementSpeed * System::Time::deltaTime;
    if (glfwGetKey(System::Window::window, GLFW_KEY_A) == GLFW_PRESS)
        camera->transform->position -= camera->Right * movementSpeed * System::Time::deltaTime;
    if (glfwGetKey(System::Window::window, GLFW_KEY_D) == GLFW_PRESS)
        camera->transform->position += camera->Right * movementSpeed * System::Time::deltaTime;
    if (glfwGetKey(System::Window::window, GLFW_KEY_Q) == GLFW_PRESS)
        camera->transform->position += camera->Up * movementSpeed * System::Time::deltaTime;
    if (glfwGetKey(System::Window::window, GLFW_KEY_E) == GLFW_PRESS)
        camera->transform->position -= camera->Up * movementSpeed * System::Time::deltaTime;

    yaw += (System::Input::mouse.xPosition - xMousePos) * mouseSensitivity;
    pitch -= (yMousePos - System::Input::mouse.yPosition) * mouseSensitivity;

    xMousePos = System::Input::mouse.xPosition;
    yMousePos = System::Input::mouse.yPosition;

    if (pitch > 89.0f)
        pitch = 89.0f;

    if (pitch < -89.0f)
        pitch = -89.0f;

    camera->transform->rotateOfEulerAngles(Athena::Vector3(0, yaw, 0));

    camera->Front = Athena::Vector3(std::cos(yaw), 0, -std::sin(yaw));
    camera->Right = camera->Front.cross(Athena::Vector3::up()).normalized();
    // camera->Up = camera->Front.cross(camera->Right).normalized();
}

void CameraMovement::setOrderOfExecution(const short& newOrderOfExecution)
{
    _orderOfExecution = newOrderOfExecution;
}

short CameraMovement::getUniqueID()
{
    return 12;
}

std::string CameraMovement::toString()
{
    return "Camera Movement";
}

CameraMovement::~CameraMovement()
{
    delete camera;
}
