#ifndef COLLISIONPLANE_HPP
#define COLLISIONPLANE_HPP

namespace Khronos
{
    // Planes are always associated with immovagle geometry so the 
    // rigid body pointer in the Primitive class will be null
    class CollisionPlane
    {
        public:

            // Holds the plane's normal
            Athena::Vector3 direction;
            
            // Holds the distance of the plane from the world origin
            Athena::Scalar offset;

            CollisionPlane(const Athena::Vector3& direction, const Athena::Scalar offset)
            : direction(direction), offset(offset)
            {}
    };
}

#endif