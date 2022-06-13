#ifndef VECTOR4_HPP
#define VECTOR4_HPP

#include <Vector.hpp>
#include <Vector3.hpp>

namespace Athena 
{
    typedef float Scalar;

    template<typename K>
    struct Vector4Coordinates
    {
        K x;
        K y;
        K z;
        K w;
    };

    class Vector4 : Vector<Vector4Coordinates<Scalar>, Vector4, Scalar>
    {
        public:
            Vector4Coordinates<Scalar> coordinates;

            Vector4();
            Vector4(const Scalar& x, const Scalar& y, const Scalar& z, const Scalar& w);
            Vector4(const Vector2& vector, const Scalar& z, const Scalar& w);
            Vector4(const Vector3& vector, const Scalar& w);
            Vector4(const Vector4& vector);

            Scalar dot(const Vector4& vector) const;

            Scalar magnitude() const;
            Scalar squareMagnitude() const;

            Vector4 normalized() const;
            void normalize();

            Scalar operator [] (const short& i) const;
            Scalar& operator [] (const short& i);
        
            Scalar operator * (const Vector4& vector) const;
            Vector4 operator + (const Vector4& vector) const;
            Vector4 operator - (const Vector4& vector) const;
            bool operator == (const Vector4& vector) const;

            Vector4 operator - () const;
            Vector4 operator * (const Scalar& k) const;
            Vector4 operator / (const Scalar& k) const;

            void operator += (const Vector4& vector);
            void operator -= (const Vector4& vector);
            void operator *= (const Scalar& k);
            void operator /= (const Scalar& k);

            Scalar angleBetween (const Vector4& vector) const;

            Vector4 lerp(const Vector4& vector, const Scalar& t) const;

            bool isZero() const;
            bool areEquals(const Vector4& vector) const;

            void print() const;
    };
}

#endif