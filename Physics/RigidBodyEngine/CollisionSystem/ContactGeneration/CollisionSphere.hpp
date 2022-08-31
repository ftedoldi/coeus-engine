#ifndef COLLISIONSPHERE_HPP
#define COLLISIONSPHERE_HPP
//#include <CollisionPrimitive.hpp>

namespace Khronos
{
    class CollisionPrimitive;
    
    class CollisionSphere : public CollisionPrimitive
    {
    public:
        /**
         * The data we need for a sphere is its center and radius.
         * The center is given by the offset from the origin of the
         * rigid body.
         * The radius needs to be given.
        */

       Athena::Scalar radius;

       CollisionSphere(const Athena::Scalar radius)
       {
            this->radius = radius;
       }
    };
}

#endif