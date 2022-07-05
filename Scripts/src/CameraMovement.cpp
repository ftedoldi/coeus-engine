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
}

void CameraMovement::update()
{
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

    Athena::Vector3 direction;
    direction.coordinates.x = std::cos(Athena::Math::degreeToRandiansAngle(yaw)) * std::cos(Athena::Math::degreeToRandiansAngle(pitch));
    direction.coordinates.y = std::sin(Athena::Math::degreeToRandiansAngle(pitch));
    direction.coordinates.z = std::sin(Athena::Math::degreeToRandiansAngle(yaw)) * std::cos(Athena::Math::degreeToRandiansAngle(pitch));

    Athena::Vector3 axis = Athena::Vector3::cross(camera->Front, direction).normalized();
    camera->transform->rotate(Athena::Quaternion::EulerAnglesToQuaternion(Athena::Vector3(pitch, yaw, 0)));

    // camera->Front = direction.normalized();
    // camera->Right = camera->Front.cross(Athena::Vector3::up()).normalized();
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
