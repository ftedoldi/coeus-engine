#ifndef CAMERA_HPP
#define CAMERA_HPP

#include <Component.hpp>

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

    class Camera : public Component
    {
        public:
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
            Camera();
            // Camera(Athena::Vector3 up = Athena::Vector3::up(), float yaw = YAW, float pitch = PITCH);
            // Camera(float upX, float upY, float upZ, float yaw, float pitch);

            static Athena::Matrix4 lookAt(const Athena::Vector3& position, const Athena::Vector3& forward, const Athena::Vector3& up);

            static Athena::Matrix4 perspective(const float& fieldOfView, const float& aspectRatio, const float& nearPlane, const float& farPlane);

            Athena::Matrix4 GetViewMatrix() const;

            virtual void start();
            virtual void update();

            virtual void setOrderOfExecution(const short& newOrderOfExecution);

            virtual short getUniqueID();

            virtual std::string toString();

        private:

        void updateCameraVectors();

    };
}
#endif