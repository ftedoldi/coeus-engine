#pragma once 

#ifndef QUATERNION_HPP
#define QUATERNION_HPP

#include <Scalar.hpp>
#include <Vector3.hpp>
#include <Vector4.hpp>
#include <Matrix3.hpp>
#include <Math.hpp>

#include <iostream>
#include <cmath>

#include <stdexcept>

namespace Athena
{
    class Vector4;

    class Quaternion
    {
        private:
            Vector3 _immaginary;
            Scalar _real;
        
        public:
            const Vector3& immaginary;
            const Scalar& real;

            Quaternion();
            Quaternion(const Scalar& a, const Scalar& b, const Scalar& c, const Scalar& d);
            Quaternion(const Vector3& immaginary, const Scalar& real);
            Quaternion(const Vector4& quaternion);
            Quaternion(const Quaternion& quaternion);

            static Quaternion Identity();

            static Quaternion AxisAngleToQuaternion(const Degree& angle, const Vector3& axis);
            
            static void QuaternionToAxisAngle(const Quaternion& quaternion, Scalar& angle, Vector3& axis);

            static Quaternion EulerAnglesToQuaternion(const Vector3& eulerAngles);

            static Vector3 QuaternionToEulerAngles(const Quaternion& quaternion);

            static Quaternion Matrix3ToQuaternion(const Matrix3& matrix);

            static Matrix3 QuaternionToMatrx3(const Quaternion& quaternion);

            static Vector4 AsVector4(const Quaternion& quaternion);

            Quaternion fromMatrix(const Matrix3& matrix) const;
            Quaternion fromEulerAngles(const Vector3& eulerAngles) const;
            Quaternion fromAxisAngle(const Degree& angle, const Vector3& axis) const;

            Matrix3 toMatrix3() const;
            Vector3 toEulerAngles() const;
            void toAxisAngle(Degree& angle, Vector3& axis) const;

            void conjugate();
            Quaternion conjugated() const;

            Scalar squareMagnitude() const;
            Scalar magnitude() const;

            void invert();
            Quaternion inverse() const;

            Quaternion operator /(const Scalar& k) const;
            Quaternion operator *(const Scalar& k) const;

            void operator /=(const Scalar& k);
            void operator *=(const Scalar& k);
            void operator +=(const Quaternion& quaternion);
            void operator -=(const Quaternion& quaternion);

            bool operator ==(const Quaternion& quaternion) const;
            bool operator !=(const Quaternion& quaternion) const;

            // Rotate the vector passed as input by this quaternion
            Vector3 rotateVectorByThisQuaternion(const Vector3& vectorToRotate);

            Vector4 asVector4() const;

            Quaternion normalized() const;

            void print() const;
    };

    Quaternion operator *(const Quaternion& a, const Quaternion& b);
}

#endif