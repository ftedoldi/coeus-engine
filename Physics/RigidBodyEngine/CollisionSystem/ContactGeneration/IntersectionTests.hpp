#ifndef INTERSECTIONTEST_HPP
#define INTERSECTIONTEST_HPP
#include <CollisionPrimitive.hpp>
#include <CollisionSphere.hpp>
#include <CollisionPlane.hpp>
#include <CollisionBox.hpp>
#include <CollisionUtility.hpp>

namespace Khronos
{
    /**
     * Class that holds fast intersection tests.
     * This tests can be used to early out in the
     * collision system
    */
    class CollisionDetector;

    class IntersectionTests
    {
        friend class CollisionDetector;

        static bool sphereAndSphere(const CollisionSphere& firstSphere, const CollisionSphere& secondSphere);

        static bool sphereAndHalfSpace(const CollisionSphere& sphere, const CollisionPlane& plane);

        static bool boxAndHalfSpace(const CollisionBox& box, const CollisionPlane& plane);
    };
}

#endif