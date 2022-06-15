#ifndef QUATERNION_HPP
#define QUATERNION_HPP

#include <Scalar.hpp>
#include <Vector3.hpp>
#include <Vector4.hpp>
#include <Matrix3.hpp>
#include <Math.hpp>

#include <assert.h>

#include <iostream>
#include <cmath>

#include <stdexcept>

namespace Athena
{
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

            static Quaternion Identity() { return Quaternion(0, 0, 0, 1); }

            static Quaternion AxisAngleToQuaternion(const Degree& angle, const Vector3& axis) {
                Scalar angleRad = Math::degreeToRandiansAngle(angle);

                return Quaternion(
                    axis * std::sin(angleRad * 0.5f),
                    std::cos(angleRad * 0.5f)
                );
            }
            
            static void QuaternionToAxisAngle(const Quaternion& quaternion, Scalar& angle, Vector3& axis) {
                Vector3 immm = quaternion.getImmaginaryPart();
                Scalar real = quaternion.getRealPart();

                auto im = immm.coordinates;

                angle = 2.f * std::acos(real);

                axis.coordinates.x = im.x / std::sqrt(1.f - real * real);
                axis.coordinates.y = im.y / std::sqrt(1.f - real * real);
                axis.coordinates.z = im.z / std::sqrt(1.f - real * real);
            }

            static Quaternion EulerAnglesToQuaternion(const Vector3& eulerAngles) {
                Vector3 angles = Vector3(eulerAngles);

                angles.coordinates.x = Math::degreeToRandiansAngle(eulerAngles.coordinates.x);
                angles.coordinates.y = Math::degreeToRandiansAngle(eulerAngles.coordinates.y);
                angles.coordinates.z = Math::degreeToRandiansAngle(eulerAngles.coordinates.z);

                auto pos = angles.coordinates;

                float cy = std::cos(pos.z * 0.5);
                float sy = std::sin(pos.z * 0.5);
                float cr = std::cos(pos.y * 0.5);
                float sr = std::sin(pos.y * 0.5);
                float cp = std::cos(pos.x * 0.5);
                float sp = std::sin(pos.x * 0.5);

                return Quaternion(
                    cy * cr * sp + sy * sr * cp,
                    cy * sr * cp - sy * cr * sp,
                    cy * sr * cp - sy * cr * sp,
                    cy * cr * cp + sy * sr * sp
                );
            }

            static Vector3 QuaternionToEulerAngles(const Quaternion& quaternion) {
                Vector3 result = Vector3();
                Vector4 quat = Vector4(quaternion.getImmaginaryPart(), quaternion.getRealPart());

                result.print();

                auto q = quat.coordinates;
                auto pos = result.coordinates;

                pos.y = Math::radiansToDegreeAngle(std::atan2(2.f * q.x * q.w + 2.f * q.y * q.z, 1 - 2.f * (q.z * q.z + q.w * q.w)));
                pos.x = Math::radiansToDegreeAngle(std::asin(2.f * (q.x * q.z - q.w * q.y)));
                pos.z = Math::radiansToDegreeAngle(std::atan2(2.f * q.x * q.y + 2.f * q.z * q.w, 1 - 2.f * (q.y * q.y + q.z * q.z)));

                pos.x = pos.x > 360.f ? pos.x - 360.f : pos.x;
                pos.x = pos.x < 0.f ? pos.x + 360.f : pos.x;
                pos.y = pos.y > 360.f ? pos.y - 360.f : pos.y;
                pos.y = pos.y < 0.f ? pos.y + 360.f : pos.y;
                pos.z = pos.z > 360.f ? pos.z - 360.f : pos.z;
                pos.z = pos.z < 0.f ? pos.z + 360.f : pos.z;

                return result;
            }

            static Quaternion Matrix3ToQuaternion(const Matrix3& matrix) {
                auto vecMatrix = matrix.asVector3Array();

                auto r0 = vecMatrix.row0;
                auto r1 = vecMatrix.row1;
                auto r2 = vecMatrix.row2;

                float realPart = std::sqrt(1 + r0[0] + r1[1] + r2[2]) / 2.f;

                return Quaternion(
                    (r2[1] - r1[2]) / (4.f * realPart),
                    (r0[2] - r2[0]) / (4.f * realPart),
                    (r1[0] - r0[1]) / (4.f * realPart),
                    realPart
                );
            }

            static Matrix3 QuaternionToMatrx3(const Quaternion& quaternion) {
                Vector4* quat = Quaternion::AsVector4(quaternion);

                auto q = quat->coordinates;

                return Matrix3 (
                    1 - 2.f * q.y * q.y - 2.f * q.z * q.z, 2.f * q.x * q.y - 2.f * q.z * q.w, 2.f * q.x * q.z + 2.f * q.y * q.w,
                    2.f * q.x * q.y + 2.f * q.z * q.w, 1 - 2.f * q.x  * q.x - 2.f * q.z * q.z, 2.f * q.y * q.z - 2.f * q.x * q.w,
                    2.f * q.x * q.z - 2.f * q.y * q.w, 2.f * q.y * q.z + 2.f * q.x * q.w, 1 - 2.f * q.x * q.x - 2.f * q.y * q.y
                );
            }

            static Vector4* AsVector4(const Quaternion& quaternion) {
                return new Vector4(quaternion.getImmaginaryPart(), quaternion.getRealPart());
            }

            Vector3 getImmaginaryPart() const;
            Scalar getRealPart() const;

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

            Vector4* asVector4() const;

            void print() const;

            static void TestClass() {
                assert(Quaternion() == Quaternion::Identity());
                //assert(Quaternion().asVector4() == Quaternion::Identity().asVector4());
            }
    };

    Quaternion operator *(const Quaternion& a, const Quaternion& b);
}

#endif