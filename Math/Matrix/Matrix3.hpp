#ifndef MATRIX3_HPP
#define MATRIX3_HPP
#include "Matrix.hpp"
#include "Matrix2.hpp"
#include "../Scalar.hpp"

namespace Athena 
{
    class Matrix2;

    class Matrix3 : public Matrix<3, 3, Matrix3, Scalar>
    {
    public:

        Matrix3();
        Matrix3(const Matrix2& mat);
        Matrix3(const Matrix3& mat);
        Matrix3(Scalar v1, Scalar v2, Scalar v3, Scalar v4, Scalar v5, Scalar v6, Scalar v7, Scalar v8, Scalar v9);
        
        Matrix3 operator*(const Matrix3& mat);
        Matrix3 operator*(const Scalar value);
        Vector3 operator*(const Vector3& vec);
        Matrix3 operator-(const Matrix3& mat);
        Matrix3 operator+(const Matrix3& mat);
        Matrix3 operator-();

        void operator+=(const Matrix3& mat);
        void operator*=(const Matrix3& mat);
        void operator*=(const Scalar value);
        void operator-=(const Matrix3& mat);

        bool operator==(const Matrix3& mat);
        
        void setInverse(const Matrix3& mat);
        Matrix3 inverse() const;
        static Matrix3 inverse(const Matrix3& mat);

        void setTranspose(const Matrix3& mat);
        Matrix3 transpose();
        static Matrix3 transpose(const Matrix3& mat);

        void print() const;

    };
}

#endif