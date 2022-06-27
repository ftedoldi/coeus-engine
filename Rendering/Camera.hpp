#ifndef CAMERA_HPP
#define CAMERA_HPP

#include <coeus.hpp>
#include <glm/glm.hpp>

#include <vector>

using namespace Athena;

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

    float yValue = 0.0f;

    class Camera
    {
    public:
        Vector3 Position;
        Vector3 Front;
        Vector3 Right;
        Vector3 WorldUp;
        Vector3 Up;

        float Yaw;
        float Pitch;

        float MovementSpeed;
        float MouseSensitivity;

        //costructor with vectors
        Camera(Vector3 position = Vector3(0.0f, 0.0f, 0.0f), Vector3 up = Vector3(0.0f, 1.0f, 0.0f), float yaw = YAW, float pitch = PITCH) : Front(Vector3(0.0f, 0.0f, -1.0f)), MovementSpeed(SPEED), MouseSensitivity(SENSITIVITY)
        {
            Position = position;
            WorldUp = up;
            Yaw = yaw;
            Pitch = pitch;
            updateCameraVectors();
        }

        Camera(float posX, float posY, float posZ, float upX, float upY, float upZ, float yaw, float pitch) : Front(Vector3(0.0f, 0.0f, -1.0f)), MovementSpeed(SPEED), MouseSensitivity(SENSITIVITY)
        {
            Position = Vector3(posX, posY, posZ);
            WorldUp = Vector3(upX, upY, upZ);
            Yaw = yaw;
            Pitch = pitch;
            updateCameraVectors();
        }

        void ProcessKeyboard(Camera_Movement direction, float deltaTime)
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

        void ProcessMouseMovement(float xoffset, float yoffset)
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

        Matrix4 GetViewMatrix()
        {
            return Matrix4::lookAt(this->Position, this->Position + this->Front, this->Up);
        }

    private:

    void updateCameraVectors()
    {
        Vector3 direction;
        direction.coordinates.x = std::cos(Math::degreeToRandiansAngle(Yaw)) * std::cos(Math::degreeToRandiansAngle(Pitch));
        direction.coordinates.y = std::sin(Math::degreeToRandiansAngle(Pitch));
        direction.coordinates.z = std::sin(Math::degreeToRandiansAngle(Yaw)) * std::cos(Math::degreeToRandiansAngle(Pitch));
        this->Front = Vector3::normalize(direction);

        this->Right = Vector3::normalize(Vector3::cross(this->Front, this->WorldUp));
        this->Up = Vector3::normalize(Vector3::cross(this->Right, this->Front));
    }

    };
}
#endif