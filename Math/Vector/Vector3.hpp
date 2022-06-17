#ifndef VECTOR3_HPP
#define VECTOR3_HPP

#include <Scalar.hpp>
#include <Vector.hpp>
#include <Vector2.hpp>
#include <Math.hpp>

#include <stdexcept>

namespace Athena
{
    template<typename K>
    struct Vector3Coordinates
    {
        K x;
        K y;
        K z;
    };

    class Vector2;

    class Vector3
    {
        public:
            Vector3Coordinates<Scalar> coordinates;

            Vector3();
            Vector3(const Scalar& x, const Scalar& y, const Scalar& z);
            Vector3(const Vector2& vector, const Scalar& z);
            Vector3(const Vector3& vector);
        
            static Vector3 cross(const Vector3& vector1, const Vector3& vector2);

            static Scalar dot(const Vector3& vector1, const Vector3& vector2);

            static Vector3 up();
            static Vector3 down();
            static Vector3 right();
            static Vector3 left();
            static Vector3 forward();
            static Vector3 backward();

            Vector3 cross(const Vector3& vector) const;
            Scalar dot(const Vector3& vector) const;

            Scalar magnitude() const;
            Scalar squareMagnitude() const;
            Vector3 normalized() const;
            void normalize();
            static Vector3 normalize(const Vector3& vec);

            Scalar operator [] (const short& i) const;
            Scalar& operator [] (const short& i);

            Scalar operator * (const Vector3& vector) const;
            Vector3 operator + (const Vector3& vector) const;
            Vector3 operator - (const Vector3& vector) const;
            bool operator == (const Vector3& vector) const;
            Vector3 operator - () const;
            Vector3 operator * (const Scalar& k) const;
            Vector3 operator / (const Scalar& k) const;

            void operator += (const Vector3& vector);
            void operator -= (const Vector3& vector);
            void operator *= (const Scalar& k);
            void operator /= (const Scalar& k);

            Scalar angleBetween (const Vector3& vector) const;

            Vector3 lerp(const Vector3& vector, const Scalar& t) const;
            
            bool isZero() const;

            bool areEquals(const Vector3& vector) const;
            void print() const;
    };

    Vector3 operator * (const Scalar& k, const Vector3& vector);
}

#endif