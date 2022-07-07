#include "../CameraMovement.hpp"

CameraMovement::CameraMovement()
{

}

void CameraMovement::start()
{
    this->movementSpeed = 5.0f;

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

    yaw += (System::Input::mouse.xPosition - xMousePos) * mouseSensitivity;
    pitch += (yMousePos - System::Input::mouse.yPosition) * mouseSensitivity;

    /*std::cout << "yaw: " << yaw << std::endl;
    std::cout << "pitch: " << pitch << std::endl;*/

    xMousePos = System::Input::mouse.xPosition;
    yMousePos = System::Input::mouse.yPosition;

    if (pitch > 89.0f)
        pitch = 89.0f;

    if (pitch < -89.0f)
        pitch = -89.0f;

    /*if(yaw > 360.0f)
        yaw = -360.0f;
    if(yaw < -360.0f)
        yaw = 360.0f;*/

    Athena::Vector3 direction;
    direction.coordinates.x = std::cos(Athena::Math::degreeToRandiansAngle(yaw)) * std::cos(Athena::Math::degreeToRandiansAngle(pitch));
    direction.coordinates.y = std::sin(Athena::Math::degreeToRandiansAngle(pitch));
    direction.coordinates.z = std::sin(Athena::Math::degreeToRandiansAngle(yaw)) * std::cos(Athena::Math::degreeToRandiansAngle(pitch));

    glm::vec3 newDirection(direction.coordinates.x, direction.coordinates.y, direction.coordinates.z);
    glm::vec3 oldFront = glm::normalize(newDirection);
    glm::vec3 oldRight = glm::normalize(glm::cross(oldFront, glm::vec3(0, 1, 0)));
    glm::vec3 oldUp = glm::normalize(glm::cross(oldRight, oldFront));

    /*camera->Front = Athena::Vector3::normalize(direction);
    camera->Right = Athena::Vector3::normalize(Athena::Vector3::cross(camera->Front, Athena::Vector3::up()));
    camera->Up = Athena::Vector3::normalize(Athena::Vector3::cross(camera->Right, camera->Front));*/

    camera->Front = Athena::Vector3(oldFront.x, oldFront.y, oldFront.z);
    camera->Right = Athena::Vector3(oldRight.x, oldRight.y, oldRight.z);
    camera->Up = Athena::Vector3(oldUp.x, oldUp.y, oldUp.z);


    /*Athena::Quaternion q = calculateRotation(camera->transform->position, camera->transform->position + camera->Front, camera->Up);
    Athena::Quaternion rotation = Athena::Quaternion(q.immaginary.coordinates.x, q.immaginary.coordinates.y, 0, q.real);*/
    glm::vec3 newPos((float)camera->transform->position.coordinates.x, (float)camera->transform->position.coordinates.y, (float)camera->transform->position.coordinates.z);
    glm::quat newQuat(glm::quat_cast(glm::lookAt(newPos, newPos + oldFront, oldUp)));
    Athena::Quaternion athenaQuat(newQuat.x, newQuat.y, newQuat.z, newQuat.w);

    //Athena::Quaternion quatMat(Athena::Quaternion::Matrix4ToQuaternion(camera->lookAt(camera->transform->position, camera->transform->position + camera->Front, camera->Up)));

    //Athena::Quaternion calcQuat(calculateRotation(camera->transform->position, camera->transform->position + camera->Front, camera->Up));

    camera->transform->rotate(athenaQuat);

    camera->Front = athenaQuat.rotateVectorByThisQuaternion(Athena::Vector3(0, 0, -1));
    camera->Right = Athena::Vector3::normalize(Athena::Vector3::cross(camera->Front, Athena::Vector3::up()));
    camera->Up = Athena::Vector3::normalize(Athena::Vector3::cross(camera->Front, camera->Right));


    /*camera->transform->rotateOfEulerAngles(Athena::Vector3(0, yaw, 0));

    camera->Front = Athena::Vector3(std::cos(yaw), 0, -std::sin(yaw));
    camera->Right = camera->Front.cross(Athena::Vector3::up()).normalized();*/
    // camera->Up = camera->Front.cross(camera->Right).normalized();
}

Athena::Quaternion CameraMovement::calculateRotation(const Athena::Vector3& position, const Athena::Vector3& forward, const Athena::Vector3& up) const
{
    Athena::Quaternion q = Athena::Quaternion::Identity();
    Athena::Vector3 F = Athena::Vector3::normalize(position - forward);
    Athena::Vector3 R = Athena::Vector3::normalize(Athena::Vector3::cross(up, F));
    Athena::Vector3 U = Athena::Vector3::cross(F, R);

    float trace = R.coordinates.x + U.coordinates.y + F.coordinates.z;
    if (trace > 0.0) {
        float s = 0.5 / std::sqrtf(trace + 1.0);
        q.real = 0.25 / s;
        q.immaginary.coordinates.x = (U.coordinates.z - F.coordinates.y) * s;
        q.immaginary.coordinates.y = (F.coordinates.x - R.coordinates.z) * s;
        q.immaginary.coordinates.z = (R.coordinates.y - U.coordinates.x) * s;
    } else {
    if (R.coordinates.x > U.coordinates.y && R.coordinates.x > F.coordinates.z) {
        float s = 2.0 * std::sqrtf(1.0 + R.coordinates.x - U.coordinates.y - F.coordinates.z);
        q.real = (U.coordinates.z - F.coordinates.y) / s;
        q.immaginary.coordinates.x = 0.25 * s;
        q.immaginary.coordinates.y = (U.coordinates.x + R.coordinates.y) / s;
        q.immaginary.coordinates.z = (F.coordinates.x + R.coordinates.z) / s;
    } else if (U.coordinates.y > F.coordinates.z) {
        float s = 2.0 * std::sqrtf(1.0 + U.coordinates.y - R.coordinates.x - F.coordinates.z);
        q.real = (F.coordinates.x - R.coordinates.z) / s;
        q.immaginary.coordinates.x = (U.coordinates.x + R.coordinates.y) / s;
        q.immaginary.coordinates.y = 0.25 * s;
        q.immaginary.coordinates.z = (F.coordinates.y + U.coordinates.z) / s;
    } else {
        float s = 2.0 * std::sqrtf(1.0 + F.coordinates.z - R.coordinates.x - U.coordinates.y);
        q.real = (R.coordinates.y - U.coordinates.x) / s;
        q.immaginary.coordinates.x = (F.coordinates.x + R.coordinates.z) / s;
        q.immaginary.coordinates.y = (F.coordinates.y + U.coordinates.z) / s;
        q.immaginary.coordinates.z = 0.25 * s;
    }
    }
    return q;
    
    //ANOTHER LOOKAT MATRIX IMPLEMENTATION - STILL DONT WORKS
    /*Athena::Vector3::normalize(forward);
    Athena::Vector3 F = Athena::Vector3::normalize(position - forward);
    Athena::Vector3 R = Athena::Vector3::normalize(Athena::Vector3::cross(up, F));
    Athena::Vector3 U = Athena::Vector3::cross(F, R);

    float m00 = R.coordinates.x;
    float m01 = R.coordinates.y;
    float m02 = R.coordinates.z;

    float m10 = U.coordinates.x;
    float m11 = U.coordinates.y;
    float m12 = U.coordinates.z;

    float m20 = F.coordinates.x;
    float m21 = F.coordinates.y;
    float m22 = F.coordinates.z;

    float num8 = (m00 + m11) + m22;
    Athena::Quaternion quat = Athena::Quaternion::Identity();
    if(num8 > 0.0f)
    {
        float num = (float)std::sqrt(num8 + 1.0f);
        quat.real = num * 0.5f;
        num = 0.5f / num;
        quat.immaginary.coordinates.x = (m12 - m21) * num;
        quat.immaginary.coordinates.y = (m20 - m02) * num;
        quat.immaginary.coordinates.z = (m01 - m10) * num;
        return quat;
    }

    if((m00 >= m11) && (m00 >= m22))
    {
        float num7 = (float)std::sqrt(((1.0f + m00) - m11) - m22);
        float num4 = 0.5f / num7;
        quat.immaginary.coordinates.x = 0.5f * num7;
        quat.immaginary.coordinates.y = (m01 + m10) * num4;
        quat.immaginary.coordinates.z = (m02 + m20) * num4;
        quat.real = (m12 - m21) * num4;
        return quat;
    }

    if(m11 > m22)
    {
        float num6 = (float)std::sqrt(((1.0f + m11) - m00) - m22);
        float num3 = 0.5f / num6;
        quat.immaginary.coordinates.x = (m10 + m01) * num3;
        quat.immaginary.coordinates.y = 0.5f * num6;
        quat.immaginary.coordinates.z = (m21 + m12) * num3;
        quat.real = (m20 - m02) * num3;
        return quat;
    }

    float num5 = (float)std::sqrt(((1.0f + m22) - m00) - m11);
    float num2 = 0.5f / num5;
    quat.immaginary.coordinates.x = (m20 + m02) * num2;
    quat.immaginary.coordinates.y = (m21 + m12) * num2;
    quat.immaginary.coordinates.z = 0.5f * num5;
    quat.real = (m01 - m10) * num2;
    return quat;*/
    
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
