#include "Camera.hpp"
namespace Odysseus
{
    Camera::Camera(Athena::Vector3 position, Athena::Vector3 up, float yaw, float pitch) : Front(Athena::Vector3(0.0f, 0.0f, -1.0f)), MovementSpeed(SPEED), MouseSensitivity(SENSITIVITY)
    {
        Position = position;
        WorldUp = up;
        Yaw = yaw;
        Pitch = pitch;
        updateCameraVectors();
    }

    Camera::Camera(float posX, float posY, float posZ, float upX, float upY, float upZ, float yaw, float pitch) : Front(Athena::Vector3(0.0f, 0.0f, -1.0f)), MovementSpeed(SPEED), MouseSensitivity(SENSITIVITY)
    {
        Position = Athena::Vector3(posX, posY, posZ);
        WorldUp = Athena::Vector3(upX, upY, upZ);
        Yaw = yaw;
        Pitch = pitch;
        updateCameraVectors();
    }

    void Camera::ProcessKeyboard(Camera_Movement direction, float deltaTime)
    {
        const float velocity = static_cast<float>(MovementSpeed * deltaTime);
        if(direction == FORWARD)
            Position += Front * velocity;
        if(direction == BACKWARD)
            Position -= Front * velocity;
        if(direction == LEFT)
            Position -= Right * velocity;
        if(direction == RIGHT)
            Position += Right * velocity;
        if(direction == UP)
            {
                Position += WorldUp * velocity;
                yValue = Position.coordinates.y;
            }
        if(direction == DOWN)
            {
                Position -= WorldUp * velocity;
                yValue = Position.coordinates.y;
            }
        
        Position.coordinates.y = yValue;
    }

    void Camera::ProcessMouseMovement(float xoffset, float yoffset)
    {
        xoffset *= MouseSensitivity;
        yoffset *= MouseSensitivity;

        Yaw += xoffset;
        Pitch += yoffset;

        //make sure pitch is not out of bound
        if(Pitch > 89.0f)
            Pitch = 89.0f;
        if(Pitch < -89.0f)
            Pitch = -89.0f;

        updateCameraVectors();
    }

    Athena::Matrix4 Camera::lookAt(const Athena::Vector3& position, const Athena::Vector3& forward, const Athena::Vector3& up)
    {
        Athena::Vector3 zaxis = Athena::Vector3::normalize(forward - position);
        Athena::Vector3 xaxis = Athena::Vector3::normalize(Athena::Vector3::cross(zaxis, up));
        Athena::Vector3 yaxis = Athena::Vector3::cross(xaxis, zaxis);

        Athena::Matrix4 result;
        result.data[0] = xaxis.coordinates.x;
        result.data[4] = xaxis.coordinates.y;
        result.data[8] = xaxis.coordinates.z;

        result.data[1] = yaxis.coordinates.x;
        result.data[5] = yaxis.coordinates.y;
        result.data[9] = yaxis.coordinates.z;

        result.data[2] = -zaxis.coordinates.x;
        result.data[6] = -zaxis.coordinates.y;
        result.data[10] = -zaxis.coordinates.z;

        result.data[12] = -(Athena::Vector3::dot(xaxis, position));
        result.data[13] = -(Athena::Vector3::dot(yaxis, position));
        result.data[14] = (Athena::Vector3::dot(zaxis, position));

        return result;
    }

    Athena::Matrix4 Camera::perspective(const float& fieldOfView, const float& aspectRatio, const float& nearPlane, const float& farPlane)
    {
        Athena::Matrix4 result;
        float yScale = 1.0 / std::tan((M_PI/ 180.0f) * fieldOfView / 2);
        float xScale = yScale / aspectRatio;
        result.data[0] = xScale;
        result.data[5] = yScale;
        result.data[10] = (farPlane + nearPlane) / (nearPlane - farPlane);
        result.data[11] = -1;
        result.data[14] = 2 * farPlane * nearPlane / (nearPlane - farPlane);
        result.data[15] = 0;
        return result;
    }

    Athena::Matrix4 Camera::GetViewMatrix() const
    {
        return Camera::lookAt(Position, Position + Front, Up);
    }

    void Camera::updateCameraVectors()
    {
        Athena::Vector3 direction;
        direction.coordinates.x = std::cos(Athena::Math::degreeToRandiansAngle(Yaw)) * std::cos(Athena::Math::degreeToRandiansAngle(Pitch));
        direction.coordinates.y = std::sin(Athena::Math::degreeToRandiansAngle(Pitch));
        direction.coordinates.z = std::sin(Athena::Math::degreeToRandiansAngle(Yaw)) * std::cos(Athena::Math::degreeToRandiansAngle(Pitch));
        Front = Athena::Vector3::normalize(direction);

        Right = Athena::Vector3::normalize(Athena::Vector3::cross(Front, WorldUp));
        Up = Athena::Vector3::normalize(Athena::Vector3::cross(Right, Front));
    }

}
