#ifndef TRANSFORM_HPP
#define TRANSFORM_HPP

#include <Tree.hpp>

#include <Matrix3.hpp>
#include <Matrix4.hpp>
#include <Vector3.hpp>
#include <Vector4.hpp>
#include <Quaternion.hpp>
#include <Scalar.hpp>

#include <string>
#include <memory>

namespace Odysseus
{
    namespace Zeus {
        struct Node;
        class Tree;
    }

    class Transform
    {
        private:
            Athena::Quaternion _rotation;
            Athena::Matrix4 _worldToLocalMatrix;
            Athena::Matrix4 _localToWorldMatrix;
            std::unique_ptr<Zeus::Tree> _childrenTree;

        public:
            Athena::Vector3 position;
            const Athena::Quaternion& rotation;
            Athena::Vector3 localScale;

            const Athena::Matrix4& worldToLocalMatrix;
            const Athena::Matrix4& localToWorldMatrix;

            std::string name;

            Transform(Athena::Vector3 position, Athena::Quaternion rotation, Athena::Vector3 scale);
            Transform(Athena::Vector3 position, Athena::Vector3 eulerAnglesRotation, Athena::Vector3 scale);
            Transform(Athena::Vector3 position, Athena::Vector3 eulerAnglesRotation);
            Transform(Athena::Vector3 position);
            Transform(Athena::Quaternion rotation);
            Transform();

            Transform translate(const Athena::Vector3& destination) const;
            Transform nonUniformScaleBy(const Athena::Vector3& scale) const;
            Transform uniformScaleBy(const Athena::Scalar& uniformScale) const;
            Transform rotateAroundAxis(const Athena::Vector3& axis, const Athena::Scalar& angle) const;
            Transform rotateAroundAxis(const Athena::Vector4& axisAngle) const;
            Transform rotateOfEulerAngles(const Athena::Vector3 eulerAngles) const;
            Transform rotateOfMatrix3(const Athena::Matrix3 matrix) const;
            Transform rotate(const Athena::Quaternion& rotationQuaternion) const;

            Transform lookAt(const Athena::Vector3& position) const;
            Transform lookAt(const Transform& target) const;

            void translate(const Athena::Vector3& destination);
            void nonUniformScaleBy(const Athena::Vector3& scale);
            void uniformScaleBy(const Athena::Scalar& uniformScale);
            void rotateAroundAxis(const Athena::Vector3& axis, const Athena::Scalar& angle);
            void rotateAroundAxis(const Athena::Vector4& axisAngle);
            void rotateOfEulerAngles(const Athena::Vector3 eulerAngles);
            void rotateOfMatrix3(const Athena::Matrix3 matrix);
            void rotate(const Athena::Quaternion& rotationQuaternion);

            void lookAt(const Athena::Vector3& position);
            void lookAt(const Transform& target);

            Transform transformDirection() const; // TODO Class Versor
            Transform transformVector(const Athena::Vector2& vector) const;
            Transform transformVector(const Athena::Vector3& vector) const;
            Transform transformVector(const Athena::Vector4& vector) const;
            Transform transformPoint() const; // TODO class Point
            Transform inverseTransformDirection() const; // TODO Class Versor
            Transform inverseTransformVector(const Athena::Vector2& vector) const;
            Transform inverseTransformVector(const Athena::Vector3& vector) const;
            Transform inverseTransformVector(const Athena::Vector4& vector) const;
            Transform inverseTransformPoint() const; // TODO class Point

            void addChild(Transform child);
            Zeus::Node getChild(int index);
            Zeus::Node getChild(std::string name);
            Transform getChildTransform(int index) const;
            Transform getChildTransform(std::string name) const;

            bool operator == (const Transform& b) const;
            bool operator != (const Transform& b) const;
            Transform operator * (const Transform& b) const; // Transform composition

            Transform inverse() const;

            ~Transform();
    };    
}

#endif