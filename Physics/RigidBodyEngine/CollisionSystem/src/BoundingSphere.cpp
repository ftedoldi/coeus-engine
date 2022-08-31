#include "../BoundingSphere.hpp"

namespace Khronos
{
    BoundingSphere::BoundingSphere(const Athena::Vector3& center, const Athena::Scalar radius) :
    center(center), radius(radius)
    {

    }

    BoundingSphere::BoundingSphere(const BoundingSphere& first, const BoundingSphere& second)
    {
        // Firstly we get the vector representing the offset between the two bounding sphere's center
        Athena::Vector3 centerOffset = second.center - first.center;
        // Than we get the square length of the vector (so we don't need to compute the square root)
        Athena::Scalar distance = centerOffset.squareMagnitude();
        // We get the difference between the length of the two radius
        Athena::Scalar radiusDiff = second.radius - first.radius;

        // Check if the larger sphere encloses the small one
        if(radiusDiff * radiusDiff >= distance)
        {
            if(first.radius > second.radius)
            {
                this->center = first.center;
                this->radius = first.radius;
            }
            else
            {
                this->center = second.center;
                this->radius = second.radius;
            }
        }
        // Otherwise the larger sphere doesn't enclose the small one so
        // we need to work on partially overlapping spheres
        else
        {
            distance = Athena::Math::scalarSqrt(distance);
            // We multiply by 0.5 because with only the first part of the formula we would obtain the diameter
            this->radius = (distance + first.radius + second.radius) * ((Athena::Scalar)0.5);

            // The new center is based on first's center, moved towards second's center by an amount
            // proportional to the spheres' radii
            this->center = first.center;
            if(distance > 0)
            {
                this->center += centerOffset * ((this->radius - first.radius) / distance);
            }

        }
    }

    bool BoundingSphere::overlaps(const BoundingSphere* other) const
    {
        // We use the squared distance so we don't need to compute the square root
        Athena::Scalar squaredDistance = (this->center - other->center).squareMagnitude();
        return squaredDistance < (this->radius + other->radius) * (this->radius + other->radius);
    }

    Athena::Scalar BoundingSphere::getSize() const
    {
        return ((Athena::Scalar(1.33333) * M_PI * radius * radius * radius));
    }

    Athena::Scalar BoundingSphere::getGrowth(const BoundingSphere& other)
    {
        // We create a sphere that encapsulates this sphere and other sphere
        BoundingSphere newSphere(*this, other);

        // We return a value proportional to the change in surface area of the sphere
        return newSphere.radius * newSphere.radius - radius * radius;
    }
}