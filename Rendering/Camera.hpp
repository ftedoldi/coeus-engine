#ifndef CAMERA_HPP
#define CAMERA_HPP

#include <Component.hpp>

#include <coeus.hpp>
#include <vector>

namespace Odysseus
{
    class Transform;

    class Camera : public System::Component
    {
        public:
            static Camera* main;

            Athena::Vector3& Front;
            Athena::Vector3& Right;
            Athena::Vector3& Up;

            Camera();

            static Athena::Matrix4 lookAt(const Athena::Vector3& position, const Athena::Vector3& forward, const Athena::Vector3& up);

            static Athena::Matrix4 perspective(const float& fieldOfView, const float& aspectRatio, const float& nearPlane, const float& farPlane);

            Athena::Matrix4 getViewMatrix() const;
            Transform* getViewTransform(Transform* objectTransform);

            void lookAtDirection(const Athena::Vector3& direction);
            void lookAtPitchYawDirection(const Athena::Scalar& pitch, const Athena::Scalar& yaw);
            void lookAtEulerAnglesDirection(const Athena::Vector3& eulerAngles);

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