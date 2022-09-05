#include "../CollisionPrimitive.hpp"

namespace Khronos
{
    Athena::Vector3 CollisionPrimitive::getAxis(unsigned int index) const
    {
        return transform.getAxisVector(index);
    }

    void CollisionPrimitive::calculateInternals()
    {
        this->transform = this->body->transformMatrix;
    }

}