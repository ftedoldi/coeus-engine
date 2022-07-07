#include "../CameraMovement.hpp"

CameraMovement::CameraMovement()
{

}

void CameraMovement::start()
{
    this->movementSpeed = 2.0f;

    this->yaw = -90;
    this->pitch = 0;

    this->mouseSensitivity = .2f;

    xMousePos = System::Input::mouse.xPosition;
    yMousePos = System::Input::mouse.yPosition;

    /*camera->Front = Athena::Vector3(std::cos(yaw), 0, -std::sin(yaw));
    camera->Right = camera->Front.cross(Athena::Vector3::up()).normalized();*/
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
        camera->transform->position += Athena::Vector3::up() * movementSpeed * System::Time::deltaTime;
    if (glfwGetKey(System::Window::window, GLFW_KEY_E) == GLFW_PRESS)
        camera->transform->position -= Athena::Vector3::up() * movementSpeed * System::Time::deltaTime;

    yaw -= (System::Input::mouse.xPosition - xMousePos) * mouseSensitivity;
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
    camera->Front = Athena::Vector3::normalize(direction);
    camera->Right = Athena::Vector3::normalize(Athena::Vector3::cross(camera->Front, Athena::Vector3::up()));
    camera->Up = Athena::Vector3::normalize(Athena::Vector3::cross(camera->Right, camera->Front));//camera->Front.cross(camera->Right).normalized();

    Athena::Quaternion q = calculateRotation(camera->transform->position, camera->transform->position + camera->Front, camera->Up);
    Athena::Quaternion rotation = Athena::Quaternion(q.immaginary.coordinates.x, q.immaginary.coordinates.y, 0, q.real);
    camera->transform->rotate(rotation);


    /*camera->transform->rotateOfEulerAngles(Athena::Vector3(0, yaw, 0));

    camera->Front = Athena::Vector3(std::cos(yaw), 0, -std::sin(yaw));
    camera->Right = camera->Front.cross(Athena::Vector3::up()).normalized();*/
    // camera->Up = camera->Front.cross(camera->Right).normalized();
}

Athena::Quaternion CameraMovement::calculateRotation(const Athena::Vector3& position, const Athena::Vector3& forward, const Athena::Vector3& up) const
{
    Athena::Quaternion q;
    Athena::Vector3 F = Athena::Vector3::normalize(position - forward);
    Athena::Vector3 R = Athena::Vector3::normalize(Athena::Vector3::cross(up, F));
    Athena::Vector3 U = Athena::Vector3::cross(F, R);

    double trace = R.coordinates.x + U.coordinates.y + F.coordinates.z;
    if (trace > 0.0) {
        double s = 0.5 / std::sqrt(trace + 1.0);
        q.real = 0.25 / s;
        q.immaginary.coordinates.x = (U.coordinates.z - F.coordinates.y) * s;
        q.immaginary.coordinates.y = (F.coordinates.x - R.coordinates.z) * s;
        q.immaginary.coordinates.z = (R.coordinates.y - U.coordinates.x) * s;
    } else {
    if (R.coordinates.x > U.coordinates.y && R.coordinates.x > F.coordinates.z) {
        double s = 2.0 * std::sqrt(1.0 + R.coordinates.x - U.coordinates.y - F.coordinates.z);
        q.real = (U.coordinates.z - F.coordinates.y) / s;
        q.immaginary.coordinates.x = 0.25 * s;
        q.immaginary.coordinates.y = (U.coordinates.x + R.coordinates.y) / s;
        q.immaginary.coordinates.z = (F.coordinates.x + R.coordinates.z) / s;
    } else if (U.coordinates.y > F.coordinates.z) {
        double s = 2.0 * std::sqrt(1.0 + U.coordinates.y - R.coordinates.x - F.coordinates.z);
        q.real = (F.coordinates.x - R.coordinates.z) / s;
        q.immaginary.coordinates.x = (U.coordinates.x + R.coordinates.y) / s;
        q.immaginary.coordinates.y = 0.25 * s;
        q.immaginary.coordinates.z = (F.coordinates.y + U.coordinates.z) / s;
    } else {
        double s = 2.0 * std::sqrt(1.0 + F.coordinates.z - R.coordinates.x - U.coordinates.y);
        q.real = (R.coordinates.y - U.coordinates.x) / s;
        q.immaginary.coordinates.x = (F.coordinates.x + R.coordinates.z) / s;
        q.immaginary.coordinates.y = (F.coordinates.y + U.coordinates.z) / s;
        q.immaginary.coordinates.z = 0.25 * s;
    }
    }
    return q;
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
