#ifndef BOUNDINGSPHERE_HPP
#define BOUNDINGSPHERE_HPP
#include <Vector3.hpp>

namespace Khronos
{
    class BoundingSphere
    {
        Athena::Vector3 center;
        Athena::Scalar radius;

    public:
        // Create a bounding sphere from its center and radius
        BoundingSphere(const Athena::Vector3& center, const Athena::Scalar radius);

        // Create a bounding sphere that encloses the two given bounding spheres
        BoundingSphere(const BoundingSphere& first, const BoundingSphere& second);

        // Check if this bounding sphere is overlapping with the other given one
        bool overlaps(const BoundingSphere* other) const;

        // Return the volume of the sphere
        Athena::Scalar getSize() const;

        // Return how much this bounding sphere would have grown by to incorporate the 
        // given bounding sphere.
        Athena::Scalar getGrowth(const BoundingSphere& other);

    };
}

#endif