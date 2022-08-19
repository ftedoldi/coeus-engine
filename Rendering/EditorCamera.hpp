#ifndef __EDITORCAMERA_H__
#define __EDITORCAMERA_H__

#include <Component.hpp>

#include <coeus.hpp>
#include <vector>

namespace Odysseus
{
    class Transform;

    class EditorCamera : public System::Component
    {
        public:
            Athena::Vector3& Front;
            Athena::Vector3& Right;
            Athena::Vector3& Up;

            EditorCamera();

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

            virtual int getUniqueID();

            virtual std::string toString();

            virtual ~EditorCamera() {}

        private:
            static bool _hasAlreadyAnInstance;

            Athena::Vector3 _Front;
            Athena::Vector3 _Right;
            Athena::Vector3 _Up;
    };
}

#endif // __EDITORCAMERA_H__