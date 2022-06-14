#ifndef QUATERNION_HPP
#define QUATERNION_HPP

#include <Vector3.hpp>
#include <Vector4.hpp>
#include <Matrix3.hpp>
#include <Math.hpp>

#include <iostream>
#include <cmath>

#include <stdexcept>

namespace Athena
{
    typedef float Scalar;
    typedef float Degree;

    class Quaternion
    {
        private:
            Vector3 immaginary;
            Scalar real;
        
        public:
            Quaternion();
            Quaternion(const Scalar& a, const Scalar& b, const Scalar& c, const Scalar& d);
            Quaternion(const Vector3& immaginary, const Scalar& real);
            Quaternion(const Vector4& quaternion);
            Quaternion(const Quaternion& quaternion);

            static Quaternion identity() { return Quaternion(0, 0, 0, 1); }

            static Quaternion axisAngleToQuaternion(const Degree& angle, const Vector3& axis) {
                Scalar angleRad = Math::degreeToRandiansAngle(angle);

                return Quaternion(
                    axis * std::sin(angleRad / 2),
                    std::cos(angleRad / 2)
                );
            }

            static Quaternion eulerAnglesToQuaternion(const Vector3& eulerAngles) {

            }

            static Vector3 quaternionToEulerAngles(const Quaternion& quaternion) {

            }

            Vector3 getImmaginaryPart() const;
            Scalar getRealPart() const;

            Quaternion fromMatrix(const Matrix3& matrix);
            Quaternion fromEulerAngles(const Vector3& eulerAngles);
            Quaternion fromAxisAngle(const Degree& angle, const Vector3& axis);

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

            Quaternion operator /=(const Scalar& k);
            Quaternion operator *=(const Scalar& k);
            Quaternion operator +=(const Quaternion& quaternion);
            Quaternion operator -=(const Quaternion& quaternion);

            // Rotate the vector passed as input by this quaternion
            Vector3 rotateVectorByThisQuaternion(const Vector3& vectorToRotate);

            void print() const;
    };

    Quaternion operator *(const Quaternion& a, const Quaternion& b);
}

#endif