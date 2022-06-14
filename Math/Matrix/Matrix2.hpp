#ifndef MATRIX2_HPP
#define MATRIX2_HPP

#include "../Scalar.hpp"
#include "Matrix.hpp"
#include "../Vector/Vector2.hpp"

namespace Athena 
{
    class Vector2;
    
    class Matrix2 : public Matrix<2, 2, Matrix2, Scalar>
    {
    public:

        Matrix2();
        Matrix2(Scalar v1, Scalar v2, Scalar v3, Scalar v4);
        Matrix2(const Vector2& vec1, const Vector2& vec2);
        Matrix2(const Matrix2& mat);

        Matrix2 operator*(const Matrix2& mat);
        Vector2 operator*(const Vector2& vec);
        Matrix2 operator*(const Scalar value);
        Matrix2 operator+(const Matrix2& mat);
        Matrix2 operator-(const Matrix2& mat);
        Matrix2 operator-();

        void operator*=(const Matrix2& mat);
        void operator*=(const Scalar value);
        void operator+=(const Matrix2& mat);
        void operator-=(const Matrix2& mat);

        bool operator==(const Matrix2& mat);

        void setInverse(const Matrix2& mat);
        Matrix2 inverse();
        static Matrix2 inverse(const Matrix2& mat);

        void setTranspose(const Matrix2& mat);
        Matrix2 transpose();
        static Matrix2 transpose(const Matrix2& mat);

        void print();

    };
}

#endif