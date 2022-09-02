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

    Athena::Vector3 Contact::calculateFrictionlessImpulse(Athena::Matrix3* inverseInertiaTensor)
    {
        Athena::Vector3 impulseContact;

        /**
         * Build a vector that shows the change in velocity
         * in world space for a unit impulse in the direction
         * of the contact normal
        */
        Athena::Vector3 deltaVelocityWorld = Athena::Vector3::cross(relativeContactPosition[0], contactNormal);
        // Calculate the angular acceleration of the delta velocity calculated before
        deltaVelocityWorld = inverseInertiaTensor[0] * deltaVelocityWorld;
        deltaVelocityWorld = Athena::Vector3::cross(deltaVelocityWorld, relativeContactPosition[0]);

        // Work out the change in velocity in contact coordinates
        // At the end deltaVelocity will contain the total velocity change per unit impulse
        Athena::Scalar deltaVelocity = deltaVelocityWorld * contactNormal;

        // Add the linear component of velocity change
        deltaVelocity += body[0]->getInverseMass();

        // If the second rigid body exists (e.g. we don't have a half plane)
        // we compute his change in velocity
        if(body[1] != nullptr)
        {
            Athena::Vector3 deltaVelocityWorld = Athena::Vector3::cross(relativeContactPosition[1], contactNormal);
            // Calculate the angular acceleration of the delta velocity calculated before
            deltaVelocityWorld = inverseInertiaTensor[1] * deltaVelocityWorld;
            deltaVelocityWorld = Athena::Vector3::cross(deltaVelocityWorld, relativeContactPosition[1]);

            // Add the change in velocity due to rotation
            Athena::Scalar deltaVelocity = deltaVelocityWorld * contactNormal;

            // Add the linear component of velocity change
            deltaVelocity += body[0]->getInverseMass();
        }
        /**
         * Since we are interested only on the velocity in the direction of the contact normal
         * and since we contact coordinates this direction is the X axis, we calculate only
         * change of velocity in the direction of the X axis
        */
        // For frictionless contact the impulse needed to achieve a given velocity change
        // is simply done by the desidered change in velocity divided by the impulse required.
        impulseContact.coordinates.x = desiredDeltaVelocity / deltaVelocity;
        impulseContact.coordinates.y = 0;
        impulseContact.coordinates.z = 0;

        return impulseContact;
    }

    Athena::Vector3 Contact::calculateLocalVelocity(unsigned int bodyIndex, Athena::Scalar dt)
    {
        RigidBody* thisBody = this->body[bodyIndex];
        
        // Work out the velocity of the contact point
        // Since to calculate the velocity we need both the angular velocity and linear velocity
        // firstly we compute the angular velocity with respect to the contact position
        // then we add the linear velocity.
        Athena::Vector3 velocity = Athena::Vector3::cross(thisBody->getRotation(), relativeContactPosition[bodyIndex]);
        velocity += thisBody->getVelocity();

        // Turn the velocity into contact space
        Athena::Vector3 contactVelocity = contactToWorldSpace.transpose() * velocity;

        // Calculate the amount of velocity due to only forces
        Athena::Vector3 accVelocity = thisBody->getLastFrameAcceleration() * dt;

        // Turn the acceleration velocity in contact space
        accVelocity = contactToWorldSpace.transpose() * accVelocity;

        // We ignore the component acceleration in contact normal direction
        accVelocity.coordinates.x = 0;

        // Sum the two velocities
        contactVelocity += accVelocity;

        return contactVelocity;
    }
    // Pag 342(pdf) to see how to calculate the desired velocity change with coefficent of restitution to make bounch harder

    void Contact::calculateDesiredDeltaVelocity(Athena::Scalar dt)
    {
        //TODO: awake optimization
        const static Athena::Scalar velocityLimit = (Athena::Scalar)0.25;

        // Calculate the velocity due to the acceleration force
        Athena::Scalar velocityFromAcc = 0.0;

        // Calculate the velocity in the current frame in the direction of the contact normal
        velocityFromAcc += body[0]->getLastFrameAcceleration() * dt * contactNormal;

        // Check if the second rigid body exists
        if(body[1] != nullptr)
        {
            velocityFromAcc -= body[1]->getLastFrameAcceleration() * dt * contactNormal;
        }

        // If the velocity is very slow, limit the restitution
        Athena::Scalar thisRestitution = this->restitution;
        if(Athena::Math::scalarAbs(contactVelocity.coordinates.x) < velocityLimit)
        {
            thisRestitution = (Athena::Scalar)0.0;
        }

        // Combine the bounce velocity with the removed acceleration velocity
        this->desiredDeltaVelocity = -contactVelocity.coordinates.x - 
            thisRestitution * (contactVelocity.coordinates.x - velocityFromAcc);
    }

    void Contact::applyPositionChange(Athena::Vector3 linearChange[2], Athena::Vector3 angularChange[2], Athena::Scalar penetration)
    {
        /**
         * To resolve the penetration we use the Nonlinear porjection strategy where
         * the resolution happens by using a combination of linear and angular movement.
         * We move both objects in the direction of the contact normal until they are no
         * longer interpenetrating. The movement will have a linear and angular component.
        */

        Athena::Scalar angularInertia[2];
        Athena::Scalar linearInertia[2];
        Athena::Scalar totalInertia = 0;

        Athena::Scalar angularMove[2];
        Athena::Scalar linearMove[2];

        const Athena::Scalar angularLimit = (Athena::Scalar)0.2;

        for(unsigned int i = 0; i < 2; ++i)
        {
            if(body[i] != nullptr)
            {
                Athena::Matrix3 inverseInertiaTensor = body[i]->getInverseInertiaTensorWorld();
                
                // Use the same procedure used as for calculating frictionless
                // velocity change to work out the angular inertia
                Athena::Vector3 angularInertiaWorld = Athena::Vector3::cross(relativeContactPosition[i], contactNormal);
                angularInertiaWorld = inverseInertiaTensor * angularInertiaWorld;
                angularInertiaWorld = Athena::Vector3::cross(angularInertiaWorld, relativeContactPosition[i]);
                angularInertia[i] = angularInertiaWorld * contactNormal;

                // The linear component is the inverse mass
                linearInertia[i] = body[i]->getInverseMass();

                totalInertia += linearInertia[i] + angularInertia[i];
            }
        }

        for(unsigned int i = 0; i < 2; ++i)
        {
            if(body[i] != nullptr)
            {
                Athena::Scalar sign = (i == 0)? 1 : -1;
                // Angular amount each object needs to move
                angularMove[i] = sign * this->penetration * (angularInertia[i] / totalInertia);
                // Linear amount each object needs to move
                linearMove[i] = sign * this->penetration * (linearInertia[i] / totalInertia);

                // We limit the angular move since when mass is large but inertia
                // tensor is small angular projections will be too great
                Athena::Vector3 projection = relativeContactPosition[i];
                projection.addScaledVector(contactNormal, -relativeContactPosition[i].dot(contactNormal));

                //TODO: pag 352 pdf understand
                Athena::Scalar maxMagnitude = angularLimit * projection.magnitude();

                if(angularMove[i] < -maxMagnitude)
                {
                    Athena::Scalar totalMove = angularMove[i] + linearMove[i];
                    angularMove[i] = -maxMagnitude;
                    linearMove[i] = totalMove - angularMove[i];
                }
                else if(angularMove[i] > maxMagnitude)
                {
                    Athena::Scalar totalMove = angularMove[i] + linearMove[i];
                    angularMove[i] = maxMagnitude;
                    linearMove[i] = totalMove - angularMove[i];
                }

                /**
                 * We have the linear amount of movement required by turning
                 * the rigid body (in angularMove[i]). We now need to
                 * calculate the desired rotation to achieve this.
                */
                if(angularMove[i] == 0)
                {
                    // We have no rotation
                    angularChange[i].clear();
                }
                else
                {
                    // Work out the direction we'd like to rotate in
                    Athena::Vector3 targetAngularDirection = Athena::Vector3::cross(relativeContactPosition[i], contactNormal);
                    
                    Athena::Matrix3 inverseInertiaTensor = body[i]->getInverseInertiaTensorWorld();

                    // Work out the direction we'd need to rotata to achieve that
                    angularChange[i] = (inverseInertiaTensor * targetAngularDirection) * (angularMove[i] / angularInertia[i]);
                }

                // The velocity change is just the linear movement along the contact normal
                linearChange[i] = contactNormal * linearMove[i];

                // Apply the linear movement
                Athena::Vector3 pos = body[i]->getPosition();
                pos.addScaledVector(contactNormal, linearMove[i]);
                body[i]->setPosition(pos);

                // Apply the change in orientation
                Athena::Quaternion q = body[i]->getOrientation();
                q.addScaledVector(angularChange[i], ((Athena::Scalar)1.0));
                body[i]->setOrientation(q);
            }
        }
    }

    void Contact::swapBodies()
    {
        this->contactNormal *= -1;

        RigidBody* temp = body[0];
        body[0] = body[1];
        body[1] = temp;
    }

    void Contact::calculateInternals(Athena::Scalar dt)
    {
        if(body[0] == nullptr)
            swapBodies();
        
        assert(body[0] != nullptr);

        // Calculate a set of axis at the contact point
        calculateContactBasis();

        // Store the relative position of the contact relative to each body
        this->relativeContactPosition[0] = contactPoint - body[0]->getPosition();
        if(body[1] != nullptr)
        {
            this->relativeContactPosition[1] = contactPoint - body[1]->getPosition();
        }

        // Find the relative velocity of the bodies at contact point
        this->contactVelocity = calculateLocalVelocity(0, dt);
        if(body[1] != nullptr)
        {
            this->contactVelocity -= calculateLocalVelocity(1, dt); 
        }

        // Calculate the desired change in velocity for resolution
        // based on contact velocity
        calculateDesiredDeltaVelocity(dt);
    }
}