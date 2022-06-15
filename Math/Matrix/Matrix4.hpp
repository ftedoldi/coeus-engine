#ifndef MATRIX4_HPP
#define MATRIX4_HPP

#define _USE_MATH_DEFINES
#include "../Scalar.hpp"
#include "../Vector/Vector3.hpp"
#include "../Vector/Vector4.hpp"
#include "Matrix3.hpp"

namespace Athena 
{
    class Matrix4;

    class Matrix4
    {
    public:
        Scalar data[4 * 4];

        Matrix4();
        Matrix4(const Vector4& vec1, const Vector4& vec2, const Vector4& vec3, const Vector4& vec4);
        Matrix4(const Scalar& v1, const Scalar& v2, const Scalar& v3, const Scalar& v4,
                const Scalar& v5, const Scalar& v6, const Scalar& v7, const Scalar& v8,
                const Scalar& v9, const Scalar& v10, const Scalar& v11, const Scalar& v12,
                const Scalar& v13, const Scalar& v14, const Scalar& v15, const Scalar& v16);

        void operator*=(const Matrix4& mat);
        void operator*=(const Scalar& value);
        void operator+=(const Matrix4& mat);
        void operator-=(const Matrix4& mat);

        Matrix4 operator*(const Matrix4& mat) const;
        Vector4 operator*(const Vector4& vec) const;
        Matrix4 operator*(const Scalar& value) const;
        Matrix4 operator+(const Matrix4& mat) const;
        Matrix4 operator-(const Matrix4& mat) const;
        Matrix4 operator-() const;

        void setInverse(const Matrix4& mat);
        Matrix4 inverse() const;
        static Matrix4 inverse(const Matrix4& mat);

        static Matrix4 scale(const Vector3& scale);
        static Matrix4 translate(const Vector3& translate);
        static Matrix4 lookAt(const Vector3& position, const Vector3& forward, const Vector3& up);
        static Matrix4 perspective(const float& fieldOfView, const float& nearPlane, const float& farPlane);

        void print() const;
    };
}

#endif