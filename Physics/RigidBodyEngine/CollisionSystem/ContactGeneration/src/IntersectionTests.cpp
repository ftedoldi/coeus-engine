#include "../IntersectionTests.hpp"

namespace Khronos
{
    bool IntersectionTests::sphereAndSphere(const CollisionSphere& firstSphere, const CollisionSphere& secondSphere)
    {
        // Find the vector between the spheres
        Athena::Vector3 midline = firstSphere.getAxis(3) - secondSphere.getAxis(3);

        return midline.squareMagnitude() < (firstSphere.radius + secondSphere.radius) * (firstSphere.radius + secondSphere.radius);
    }

    bool IntersectionTests::sphereAndHalfSpace(const CollisionSphere& sphere, const CollisionPlane& plane)
    {
        // Find the distance from the origin
        Athena::Scalar sphereDistance = plane.direction * sphere.getAxis(3) - sphere.radius;

        // If the distance of the sphere from the origin is greater than
        // the distance from of the plane from the origin, we don't have
        // a collision
        return sphereDistance <= plane.offset;
    }

    bool IntersectionTests::boxAndHalfSpace(const CollisionBox& box, const CollisionPlane& plane)
    {
        Athena::Scalar projectedRadius = CollisionUtility::transformToAxis(box, plane.direction);

        // Work out how far is the box from the origin
        Athena::Scalar boxDistance = plane.direction * box.getAxis(3) - projectedRadius;

        // Check for intersections
        return boxDistance <= plane.offset;
    }
}