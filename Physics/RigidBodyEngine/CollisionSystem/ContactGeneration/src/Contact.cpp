#include "../Contact.hpp"

namespace Khronos
{
    void Contact::calculateContactBasis()
    {
        Athena::Vector3 contactTangent[2];

        /**
         * Check wheter the Z-axis is nearer to the X or Y axis
         * The maximum absolute value amoung X or Y will give the
         * closest axis
        */
        if(Athena::Math::scalarAbs(contactNormal.coordinates.x) > Athena::Math::scalarAbs(contactNormal.coordinates.y))
        {
            // Scaling factor to make sure the results are always normalized
            const Athena::Scalar s = (Athena::Scalar)1.0 / Athena::Math::scalarSqrt(
                contactNormal.coordinates.z * contactNormal.coordinates.z +
                contactNormal.coordinates.x * contactNormal.coordinates.x);
            
            contactTangent[0].coordinates.x = contactNormal.coordinates.z * s;
            contactTangent[0].coordinates.y = 0;
            contactTangent[0].coordinates.z = -contactNormal.coordinates.x * s;

            // Vector product in place for optimization
            contactTangent[1].coordinates.x = contactNormal.coordinates.y * contactTangent[0].coordinates.x;
            contactTangent[1].coordinates.y = contactNormal.coordinates.z * contactTangent[0].coordinates.x -
                contactNormal.coordinates.x * contactTangent[0].coordinates.z;
            contactTangent[1].coordinates.z = -contactNormal.coordinates.y * contactTangent[0].coordinates.x;
        }
        else
        {
            // The Y axis is closest to the Z axis
            const Athena::Scalar s = (Athena::Scalar)1.0 / Athena::Math::scalarSqrt(
                contactNormal.coordinates.z * contactNormal.coordinates.z +
                contactNormal.coordinates.y * contactNormal.coordinates.y);

            contactTangent[0].coordinates.x = 0;
            contactTangent[0].coordinates.y = -contactNormal.coordinates.z * s;
            contactTangent[0].coordinates.z = contactNormal.coordinates.y * s;

            contactTangent[1].coordinates.x = contactNormal.coordinates.y * contactTangent[0].coordinates.z -
                contactNormal.coordinates.z * contactTangent[0].coordinates.y;
            contactTangent[1].coordinates.y = -contactNormal.coordinates.x * contactTangent[0].coordinates.z;
            contactTangent[1].coordinates.z = contactNormal.coordinates.x * contactTangent[0].coordinates.y;
        }

        contactToWorldSpace.setComponents(contactNormal, contactTangent[0], contactTangent[1]);
    }
}