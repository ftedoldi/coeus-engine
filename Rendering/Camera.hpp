#ifndef CAMERA_HPP
#define CAMERA_HPP

#include <Component.hpp>

#include <coeus.hpp>
#include <vector>

namespace Odysseus
{
    // enum Camera_Movement
    // {
    //     FORWARD,
    //     BACKWARD,
    //     LEFT,
    //     RIGHT,
    //     UP,
    //     DOWN
    // };

    // const float YAW = -90.0f;
    // const float PITCH = 0.0f;
    // const float SPEED = 2.5f;
    // const float SENSITIVITY = 0.1f;

    class Camera : public Component
    {
        public:
            const Athena::Vector3& Front;
            const Athena::Vector3& Right;
            const Athena::Vector3& Up;

            Camera();

            static Athena::Matrix4 lookAt(const Athena::Vector3& position, const Athena::Vector3& forward, const Athena::Vector3& up);

            static Athena::Matrix4 perspective(const float& fieldOfView, const float& aspectRatio, const float& nearPlane, const float& farPlane);

            Athena::Matrix4 GetViewMatrix() const;
            Odysseus::Transform* GetViewTransform(Odysseus::Transform* objectTransform) const;

            virtual void start();
            virtual void update();

            virtual void setOrderOfExecution(const short& newOrderOfExecution);

            virtual short getUniqueID();

            virtual std::string toString();

        private:
            Athena::Vector3 _Front;
            Athena::Vector3 _Right;
            Athena::Vector3 _Up;
    };
}
#endif