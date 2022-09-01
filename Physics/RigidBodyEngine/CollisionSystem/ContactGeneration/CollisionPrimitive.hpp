#ifndef COLLISIONPRIMITIVE_HPP
#define COLLISIONPRIMITIVE_HPP
#include <RigidBody.hpp>
//#include <CollisionDetector.hpp>
//#include <IntersectionTests.hpp>
//#include <CollisionUtility.hpp>

namespace Khronos
{
    class CollisionDetector;
    class IntersectionTests;
    class CollisionUtility;
    // Class that represents a primitive to detect collision against
    class CollisionPrimitive
    {
    public:
        // These classes needs to access the private and protected variables/methods
        // of this primitive class
        friend class IntersectionTests;
        friend class CollisionDetector;
        friend class CollisionUtility;

        // The rigid body that represents this primitive
        RigidBody* body;

        // Holds the offset of this primitive (cube, sphere, ecc.) from the given rigid body
        Athena::Matrix4 offset;

        // Function that allow us to get the column vector by the given index
        Athena::Vector3 getAxis(unsigned int index) const;

    protected:
        /**
         * Returns the resultant transform of the primitive
         * calculated from the combined offset of the primitive
         * and the transform matrix of the rigid body to which
         * it is attached
        */
        Athena::Matrix4 transform;
    };
}

#endif