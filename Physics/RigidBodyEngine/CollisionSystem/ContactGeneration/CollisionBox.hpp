#ifndef COLLISIONBOX_HPP
#define COLLISIONBOX_HPP
#include <CollisionPrimitive.hpp>

namespace Khronos
{
    class CollisionBox : public CollisionPrimitive
    {
        public:
            Athena::Vector3 halfSize;

            CollisionBox(const Athena::Vector3& halfSize)
            :halfSize(halfSize)
            {}
    };
}

#endif