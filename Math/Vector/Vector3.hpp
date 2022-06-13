#ifndef VECTOR3_HPP
#define VECTOR3_HPP

#include <Vector.hpp>
#include <Vector2.hpp>

namespace Athena
{
    typedef float Scalar;

    template<typename K>
    struct Vector3Coordinates
    {
        K x;
        K y;
        K z;
    };

    class Vector2;

    class Vector3 : Vector<Vector3Coordinates<Scalar>, Vector3, Scalar>
    {
        public:
            Vector3Coordinates<Scalar> coordinates;

            Vector3();
            Vector3(const Scalar& x, const Scalar& y, const Scalar& z);
            Vector3(const Vector2& vector, const Scalar& z);
            Vector3(const Vector3& vector);
        
            static Vector3 cross(const Vector3& vector1, const Vector3& vector2) {
                return Vector3(
                    vector1.coordinates.y * vector2.coordinates.z - vector1.coordinates.z * vector2.coordinates.y,
                    vector1.coordinates.z * vector2.coordinates.x - vector1.coordinates.z * vector2.coordinates.z,
                    vector1.coordinates.x * vector2.coordinates.y - vector1.coordinates.y * vector2.coordinates.x
                );
            }

            Vector3 cross(const Vector3& vector) const;
            Scalar dot(const Vector3& vector) const;

            Scalar magnitude() const;
            Scalar squareMagnitude() const;
            Vector3 normalized() const;
            void normalize();

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
}

#endif