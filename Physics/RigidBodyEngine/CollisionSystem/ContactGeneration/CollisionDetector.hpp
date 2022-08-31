#ifndef COLLISIONDETECTOR_HPP
#define COLLISIONDETECTOR_HPP
//#include <IntersectionTests.hpp>
#include <CollisionPrimitive.hpp>
#include <CollisionSphere.hpp>
#include <CollisionPlane.hpp>
#include <CollisionBox.hpp>
#include <CollisionData.hpp>

namespace Khronos
{
    class CollisionDetector
    {
    public:

        static unsigned int sphereAndSphere(const CollisionSphere& firstSphere, const CollisionSphere& secondSphere, CollisionData* data);

        /**
         * Does a collision test on a collision sphere and a plane representing
         * a half-space (i.e. the normal of the plane
         * points out of the half-space).
         */
        static unsigned int sphereAndHalfSpace(const CollisionSphere& sphere, const CollisionPlane& plane, CollisionData* data);

        /**
         * Box and half space collision is different from the sphere collision.
         * Since we are trying to use contacts that are as simple to process as
         * possible, we prefer to use point-face contacts.
         * In the case of a box is possible and instead of returning the contact
         * of a face of the box with the half space, we return four contacts
         * for each corner point of the box.
         * If an edge collides with the plane, we treat it as two point-face contact.
        */
        static unsigned int boxAndHalfSpace(const CollisionBox& box, const CollisionPlane& plane, CollisionData* data);

        /**
         * 
        */
        static unsigned int boxAndSphere(const CollisionBox& box, const CollisionSphere& sphere, CollisionData* data);

        /**
         * This type of collision is used in the box box collision as the collision detection
         * between a point of a box and a face of the other
        */
        static unsigned int boxAndPoint(const CollisionBox& box, const Athena::Vector3& point, CollisionData* data);
    };
}

#endif