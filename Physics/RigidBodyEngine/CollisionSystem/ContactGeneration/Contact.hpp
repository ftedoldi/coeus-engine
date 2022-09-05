#ifndef CONTACT_HPP
#define CONTACT_HPP
#include <Vector3.hpp>
#include <RigidBody.hpp>

namespace Khronos
{
    class ContactResolver;
    /**
     * The contact class represents two bodies in contact.
     * It has no callable functions because it holds only
     * the contact details
    */
    class Contact
    {
        friend class ContactResolver;
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

        void clear();

        protected:

        /**
         * Transform matrix that convert coordinates in contact
         * space to coordinates in world space.
        */
        Athena::Matrix3 contactToWorldSpace;

        /**
         * Holds the closing velocity at the point of contact
        */
        Athena::Vector3 contactVelocity;

        /**
         * Holds the world space position of the contact point of the two
         * objects colliding, reletive to their center.
         * This is set when the calculateInternals function is run.
        */
        Athena::Vector3 relativeContactPosition[2];

        /**
         * Holds the required change in velocity of this contact to be resolved.
        */
        Athena::Scalar desiredDeltaVelocity;

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

        Athena::Vector3 calculateFrictionlessImpulse(Athena::Matrix3* inverseInertiaTensor);

        // Calculate and return the velocity of the contact point on the given rigid body
        Athena::Vector3 calculateLocalVelocity(unsigned int bodyIndex, Athena::Scalar dt);

        void calculateDesiredDeltaVelocity(Athena::Scalar dt);

        void applyPositionChange(Athena::Vector3 linearChange[2], Athena::Vector3 angularChange[2], Athena::Scalar penetration);

        void applyVelocityChange(Athena::Vector3 velocityChange[2], Athena::Vector3 rotationChange[2]);

        /**
         * Reverses the contact. This involves swapping the two rigid bodies
         * and reversing the contact normal. This is done to make sure that 
         * if there is only one rigid body in the collision, it is at the zero
         * position of the array.
         * After calling this method the internal values must be recalculated using
         * calculateInternals method
        */
        void swapBodies();

        void calculateInternals(Athena::Scalar dt);

    };
}

#endif