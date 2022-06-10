#ifndef VECTOR3_HPP
#define VECTOR3_HPP

namespace Athena
{
    class Vector3
    {
        public:

            Vector3();
            Vector3(const Scalar& x, const Scalar& y, const Scalar& z);
            Vector3(const Vector2& vector, const Scalar& z);
            Vector3(const Vector3& vector);
        
            static Vector3 cross(const Vector3& vector1, const Vector3& vector2) {
                return Vector3();
            }

            Vector3 cross(const Vector3& vector);
    };
}

#endif