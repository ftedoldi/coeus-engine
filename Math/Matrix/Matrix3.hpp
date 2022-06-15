#ifndef MATRIX3_HPP
#define MATRIX3_HPP

#include <Vector3.hpp>
#include <Matrix.hpp>
#include <Matrix2.hpp>
#include <Scalar.hpp>

namespace Athena 
{
    struct ArrayOfVector3Matrix {
        Vector3 row0;
        Vector3 row1;
        Vector3 row2;
    };

    class Matrix2;

    class Matrix3 : public Matrix<3, 3, Matrix3, Scalar>
    {
    public:
        Matrix3();
        Matrix3(const Matrix2& mat);
        Matrix3(const Matrix3& mat);
        Matrix3(const Scalar& v1, const Scalar& v2, const Scalar& v3,
                const Scalar& v4, const Scalar& v5, const Scalar& v6,
                const Scalar& v7, const Scalar& v8, const Scalar& v9);

        static ArrayOfVector3Matrix AsVector3Array(const Matrix3& matrix) {
            ArrayOfVector3Matrix vectorMatrix;

            auto m = matrix.data;

            vectorMatrix.row0 = Vector3(m[0], m[1], m[2]);
            vectorMatrix.row1 = Vector3(m[3], m[4], m[5]);
            vectorMatrix.row2 = Vector3(m[6], m[7], m[8]);

            return vectorMatrix;
        }
        
        Matrix3 operator*(const Matrix3& mat) const;
        Matrix3 operator*(const Scalar& value) const;
        Vector3 operator*(const Vector3& vec) const;
        Matrix3 operator-(const Matrix3& mat) const;
        Matrix3 operator+(const Matrix3& mat) const;
        Matrix3 operator-() const;

        void operator+=(const Matrix3& mat);
        void operator*=(const Matrix3& mat);
        void operator*=(const Scalar& value);
        void operator-=(const Matrix3& mat);

        bool operator==(const Matrix3& mat) const;
        
        void setInverse(const Matrix3& mat);
        Matrix3 inverse() const;
        static Matrix3 inverse(const Matrix3& mat);

        void setTranspose(const Matrix3& mat);
        Matrix3 transpose() const;
        static Matrix3 transpose(const Matrix3& mat);

        ArrayOfVector3Matrix asVector3Array() const;

        void print() const;
    };
}

#endif