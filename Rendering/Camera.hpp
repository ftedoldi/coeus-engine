#ifndef CAMERA_HPP
#define CAMERA_HPP

#include <coeus.hpp>
#include <vector>

namespace Odysseus
{
    enum Camera_Movement
    {
        FORWARD,
        BACKWARD,
        LEFT,
        RIGHT,
        UP,
        DOWN
    };

    const float YAW = -90.0f;
    const float PITCH = 0.0f;
    const float SPEED = 2.5f;
    const float SENSITIVITY = 0.1f;

    class Camera
    {
    public:
        Athena::Vector3 Position;
        Athena::Vector3 Front;
        Athena::Vector3 Right;
        Athena::Vector3 WorldUp;
        Athena::Vector3 Up;
        float yValue = 0.0f;

        float Yaw;
        float Pitch;

        float MovementSpeed;
        float MouseSensitivity;

        //costructor with vectors
        Camera(Athena::Vector3 position = Athena::Vector3(0.0f, 0.0f, 0.0f), Athena::Vector3 up = Athena::Vector3(0.0f, 1.0f, 0.0f), float yaw = YAW, float pitch = PITCH);

        Camera(float posX, float posY, float posZ, float upX, float upY, float upZ, float yaw, float pitch);

        void ProcessKeyboard(Camera_Movement direction, float deltaTime);

        void ProcessMouseMovement(float xoffset, float yoffset);

        static Athena::Matrix4 lookAt(const Athena::Vector3& position, const Athena::Vector3& forward, const Athena::Vector3& up);

        static Athena::Matrix4 perspective(const float& fieldOfView, const float& aspectRatio, const float& nearPlane, const float& farPlane);

        Athena::Matrix4 GetViewMatrix() const;

    private:

    void updateCameraVectors();

    };
}
#endif