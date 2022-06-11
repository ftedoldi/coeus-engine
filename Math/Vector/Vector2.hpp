#ifndef VECTOR2_HPP
#define VECTOR2_HPP

#include "Vector.hpp"
#include "Vector3.hpp"

#include <iostream>
#include <cmath>
#include <stdexcept>

namespace Athena
{
    typedef float Scalar;

    template<typename K>
    struct Vector2Coordinates
    {
        K x;
        K y;
    };

    // Forward declaration
    class Vector2;

    class Vector2 : public Vector<Vector2Coordinates<Scalar>, Vector2, Scalar>
    {
        public:
            Vector2Coordinates<Scalar> coordinates;

            Vector2();
            Vector2(Scalar x, Scalar y);
            Vector2(const Vector2& vector);

            static Vector3 cross(const Vector2& firstVector, const Vector2& secondVector) {
                return firstVector.cross(secondVector);
            }

            Scalar dot(const Vector2& vector) const;
            Vector3 cross(const Vector2& vector) const;

            Scalar magnitude() const;
            Scalar squareMagnitude() const;
            Vector2 normalized() const;
            void normalize();

            Scalar operator [] (const short& i) const; // Readonly acess to coordinates

            Scalar& operator [] (const short& i); // Read & Write acess to coordinates

            Scalar operator * (const Vector2& vector) const;
            Vector2 operator + (const Vector2& vector) const;
            Vector2 operator - (const Vector2& vector) const;
            bool operator == (const Vector2& vector) const;

            Vector2 operator - () const;
            Vector2 operator * (const Scalar& k) const;
            Vector2 operator / (const Scalar& k) const;

            void operator += (const Vector2& vector);
            void operator -= (const Vector2& vector);
            void operator *= (const Scalar& k);
            void operator /= (const Scalar& k);

            Scalar angleBetween (const Vector2& vector) const;

            Vector2 lerp(const Vector2& vector, const Scalar& t) const;
            
            bool isZero() const;
            bool areEquals(const Vector2& vector) const;

            void print() const;
    };
}

#endif