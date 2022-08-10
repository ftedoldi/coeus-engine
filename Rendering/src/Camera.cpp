#include "..\Camera.hpp"

namespace Odysseus
{
    Camera* Camera::main;

    Camera::Camera() : Front(_Front), Right(_Right), Up(_Up)
    {
        _Front = Athena::Vector3(0, 0, -1);
        _Right = Athena::Vector3::right();
        _Up = Athena::Vector3::up();
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

    Transform* Camera::getViewTransform(Transform* objectTransform) {
        return new Transform(*this->transform->inverse() * *objectTransform);
    }

    void Camera::lookAtDirection(const Athena::Vector3& direction) {
        this->_Front = Athena::Vector3::normalize(direction);
        this->_Right = Athena::Vector3::normalize(Athena::Vector3::cross(this->Front, Athena::Vector3::up()));
        this->_Up = Athena::Vector3::normalize(Athena::Vector3::cross(this->Right, this->Front));

        Athena::Matrix4 look = this->lookAt(this->transform->position, this->transform->position + this->Front, this->Up);
        Athena::Quaternion lookAtQuat(Athena::Quaternion::matToQuatCast(look));
        
        this->transform->rotate(lookAtQuat);
    }

    void Camera::lookAtPitchYawDirection(const Athena::Scalar& pitch, const Athena::Scalar& yaw) {
        Athena::Vector3 direction;
        direction.coordinates.x = std::cos(Athena::Math::degreeToRandiansAngle(yaw)) * std::cos(Athena::Math::degreeToRandiansAngle(pitch));
        direction.coordinates.y = std::sin(Athena::Math::degreeToRandiansAngle(pitch));
        direction.coordinates.z = std::sin(Athena::Math::degreeToRandiansAngle(yaw)) * std::cos(Athena::Math::degreeToRandiansAngle(pitch));

        lookAtDirection(direction);
    }

    void Camera::lookAtEulerAnglesDirection(const Athena::Vector3& eulerAngles)
    {
        Athena::Vector3 direction;
        direction.coordinates.x = std::cos(Athena::Math::degreeToRandiansAngle(eulerAngles.coordinates.y)) * std::cos(Athena::Math::degreeToRandiansAngle(eulerAngles.coordinates.x));
        direction.coordinates.y = std::sin(Athena::Math::degreeToRandiansAngle(eulerAngles.coordinates.z));
        direction.coordinates.z = std::sin(Athena::Math::degreeToRandiansAngle(eulerAngles.coordinates.y)) * std::cos(Athena::Math::degreeToRandiansAngle(eulerAngles.coordinates.z));

        lookAtDirection(direction);
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

    Athena::Matrix4 Camera::getViewMatrix() const
    {
        return Camera::lookAt(this->transform->position, this->transform->position + Front, Up);
    }

    void Camera::start()
    {

    }

    void Camera::update()
    {

    }

    void Camera::setOrderOfExecution(const short& newOrderOfExecution)
    {
        _orderOfExecution = newOrderOfExecution;
    }

    int Camera::getUniqueID()
    {
        return this->_uniqueID;
    }

    std::string Camera::toString()
    {
        return "Camera";
    }

}
