#ifndef MATRIX3_HPP
#define MATRIX3_HPP

#include "../Scalar.hpp"
#include "../Vector/Vector3.hpp"
#include "Matrix.hpp"
#include "Matrix2.hpp"

namespace Athena 
{
    struct ArrayOfVector3Matrix {
        Vector3 row0;
        Vector3 row1;
        Vector3 row2;
    };

    class Matrix2;

    class Matrix3
    {
    public:
        Scalar data[3 * 3];
        Matrix3();
        Matrix3(const Matrix2& mat);
        Matrix3(const Matrix3& mat);
        Matrix3(Scalar v1, Scalar v2, Scalar v3, Scalar v4, Scalar v5, Scalar v6, Scalar v7, Scalar v8, Scalar v9);

        static ArrayOfVector3Matrix AsVector3Array(const Matrix3& matrix) {
            ArrayOfVector3Matrix vectorMatrix;

            auto m = matrix.data;

            vectorMatrix.row0 = Vector3(m[0], m[1], m[2]);
            vectorMatrix.row1 = Vector3(m[3], m[4], m[5]);
            vectorMatrix.row2 = Vector3(m[6], m[7], m[8]);

            return vectorMatrix;
        }
        
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

        ArrayOfVector3Matrix asVector3Array() const;

        void print() const;
    };
}

#endif