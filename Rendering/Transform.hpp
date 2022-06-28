#ifndef TRANSFORM_HPP
#define TRANSFORM_HPP

#include <Tree.hpp>

#include <Matrix3.hpp>
#include <Matrix4.hpp>
#include <Point2.hpp>
#include <Point3.hpp>
#include <Point4.hpp>
#include <Versor2.hpp>
#include <Versor3.hpp>
#include <Versor4.hpp>
#include <Vector2.hpp>
#include <Vector3.hpp>
#include <Vector4.hpp>
#include <Quaternion.hpp>
#include <Scalar.hpp>

#include <string>

namespace Zeus {
    struct Node;
    class Tree;
}

namespace Odysseus
{
    class Transform
    {
        private:
            Athena::Quaternion _rotation;
            Athena::Matrix4 _worldToLocalMatrix;
            Athena::Matrix4 _localToWorldMatrix;
            Zeus::Tree* _childrenTree;

        public:
            Athena::Vector3 position;
            const Athena::Quaternion& rotation;
            Athena::Vector3 localScale;

            const Athena::Matrix4& worldToLocalMatrix;
            const Athena::Matrix4& localToWorldMatrix;

            std::string name;

            Transform(const Athena::Vector3& pos, const Athena::Quaternion& rot, const Athena::Vector3& scale);
            Transform(const Athena::Vector3& pos, const Athena::Vector3& eulerAnglesRotation, const Athena::Vector3& scale);
            Transform(const Athena::Vector3& pos, const Athena::Vector3& eulerAnglesRotation);
            Transform(const Athena::Vector3& pos);
            Transform(const Athena::Quaternion& rot);
            Transform(const Transform& t);
            Transform();

            Zeus::Tree* childrenTree() const;

            Athena::Vector3 up() const;
            Athena::Vector3 forward() const;
            Athena::Vector3 right() const;

            Transform translate(const Athena::Vector3& destination) const;
            Transform nonUniformScaleBy(const Athena::Vector3& scale) const;
            Transform uniformScaleBy(const Athena::Scalar& uniformScale) const;
            Transform rotateAroundAxis(const Athena::Vector3& axis, const Athena::Scalar& angle) const;
            Transform rotateAroundAxis(const Athena::Vector4& axisAngle) const;
            Transform rotateOfEulerAngles(const Athena::Vector3 eulerAngles) const;
            Transform rotateOfMatrix3(const Athena::Matrix3 matrix) const;
            Transform rotate(const Athena::Quaternion& rotationQuaternion) const;

            Transform lookAt(const Athena::Vector3& pos) const;
            Transform lookAt(const Transform& target) const;

            void translate(const Athena::Vector3& destination);
            void nonUniformScaleBy(const Athena::Vector3& scale);
            void uniformScaleBy(const Athena::Scalar& uniformScale);
            void rotateAroundAxis(const Athena::Vector3& axis, const Athena::Scalar& angle);
            void rotateAroundAxis(const Athena::Vector4& axisAngle);
            void rotateOfEulerAngles(const Athena::Vector3 eulerAngles);
            void rotateOfMatrix3(const Athena::Matrix3 matrix);
            void rotate(const Athena::Quaternion& rotationQuaternion);

            void lookAt(const Athena::Vector3& pos);
            void lookAt(const Transform& target);

            Transform transformDirection(const Athena::Versor2& versor) const;
            Transform transformDirection(const Athena::Versor3& versor) const;
            Transform transformDirection(const Athena::Versor4& versor) const;
            Transform transformVector(const Athena::Vector2& vector) const;
            Transform transformVector(const Athena::Vector3& vector) const;
            Transform transformVector(const Athena::Vector4& vector) const;
            Transform transformPoint(const Athena::Point2& point) const;
            Transform transformPoint(const Athena::Point3& point) const;
            Transform transformPoint(const Athena::Point4& point) const;
            Transform inverseTransformDirection(const Athena::Versor2& versor) const;
            Transform inverseTransformDirection(const Athena::Versor3& versor) const;
            Transform inverseTransformDirection(const Athena::Versor4& versor) const;
            Transform inverseTransformVector(const Athena::Vector2& vector) const;
            Transform inverseTransformVector(const Athena::Vector3& vector) const;
            Transform inverseTransformVector(const Athena::Vector4& vector) const;
            Transform inverseTransformPoint(const Athena::Point2& point) const;
            Transform inverseTransformPoint(const Athena::Point3& point) const;
            Transform inverseTransformPoint(const Athena::Point4& point) const;

            void addChild(const Transform& child);
            Zeus::Node* getChild(const int& index);
            Zeus::Node* getChild(const std::string& name);
            Transform getChildTransform(const int& index) const;
            Transform getChildTransform(const std::string& name) const;

            bool operator == (const Transform& b) const;
            bool operator != (const Transform& b) const;
            Transform operator * (const Transform& b) const; // Transform composition

            Transform inverse() const;

            ~Transform();
    };    
}

#endif