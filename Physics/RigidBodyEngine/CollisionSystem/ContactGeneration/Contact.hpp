#ifndef CONTACT_HPP
#define CONTACT_HPP
#include <Vector3.hpp>
#include <RigidBody.hpp>

namespace Khronos
{
    /**
     * The contact class represents two bodies in contact.
     * It has no callable functions because it holds only
     * the contact details
    */
    class Contact
    {
    public:
        // Holds the bodies that are in contact
        RigidBody* body[2];

        // Holds the lateral friction coefficient at the contact
        Athena::Scalar friction;

        // Holds the normal restitution coefficient at the contact
        Athena::Scalar restitution;

        // Holds the position of the contact point in world space
        Athena::Vector3 contactPoint;

        /**
         * Holds the direction of the contact in world space.
         * Since in many cases, when a collision happens, it
         * isn't clear which direction the normal should be in
         * we use the convenction that the contact normal points
         * from the first object involved to the second.
        */
        Athena::Vector3 contactNormal;

        // Holds the depth of penetration at the contact point
        Athena::Scalar penetration;

        void setBodyData(RigidBody* first, RigidBody* second, Athena::Scalar friction, Athena::Scalar restitution);

        protected:

        /**
         * Transform matrix that convert coordinates in contact
         * space to coordinates in world space.
        */
        Athena::Matrix3 contactToWorldSpace;

        /**
         * Construct an orthonormal basis for the contact.
         * This is a system of coordinates in contact space.
         * This basis is stored inside a 3x3 matrix where
         * each column represents a vector dimension.
         * This matrix is used to transform vectors from
         * contact space to world space.
         * The X direction is generated from the contact
         * normal and the Y and Z directions are set so
         * they are at right angle to it
        */
        void calculateContactBasis();

    };
}

#endif