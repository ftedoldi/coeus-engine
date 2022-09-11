#include "../ContactResolver.hpp"

namespace Khronos
{
    ContactResolver::ContactResolver(unsigned int iterations,
                                     Athena::Scalar velocityEpsilon,
                                     Athena::Scalar positionEpsilon)
    {
        setIterations(iterations, iterations);
        setEpsilon(velocityEpsilon, positionEpsilon);
    }

    ContactResolver::ContactResolver(unsigned int velocityIterations,
                                     unsigned int positionIterations,
                                     Athena::Scalar velocityEpsilon,
                                     Athena::Scalar positionEpsilon)
    {
        setIterations(velocityIterations, positionIterations);
        setEpsilon(velocityEpsilon, positionEpsilon);
    }
        
    void ContactResolver::setIterations(unsigned int iterations)
    {
        setIterations(iterations, iterations);
    }

    
    void ContactResolver::setIterations(unsigned int velocityIterations, unsigned int positionIterations)
    {
        this->velocityIterations = velocityIterations;
        this->positionIterations = positionIterations;
    }

        
    void ContactResolver::setEpsilon(Athena::Scalar velocityEpsilon, Athena::Scalar positionEpsilon)
    {
        this->velocityEpsilon = velocityEpsilon;
        this->positionEpsilon = positionEpsilon;
    }

    bool ContactResolver::isValid()
    {
        return (velocityIterations > 0) &&
               (positionIterations > 0) &&
               (positionEpsilon >= 0.0) &&
               (velocityEpsilon >= 0.0);
    }

    void ContactResolver::prepareContacts(std::vector<Contact*>& contacts, unsigned int numContacts, Athena::Scalar dt)
    {
        // Generate contact velocity and axis informations
        for(unsigned int i = 0; i < numContacts; ++i)
        {
            // Calculate the internal contact data
            contacts.at(i)->calculateInternals(dt);
        }
    }

    void ContactResolver::adjustPositions(std::vector<Contact*>& contacts, unsigned int numContacts, Athena::Scalar dt)
    {
        // The idea is to resolve interpenetrations in order to severity

        unsigned int i, index;
        Athena::Scalar max;
        Athena::Vector3 linearChange[2], angularChange[2];
        Athena::Vector3 deltaPosition;

        this->positionIterationsUsed = 0;
        while(this->positionIterationsUsed < this->positionIterations)
        {
            // Find the biggest penetration
            max = positionEpsilon;
            index = numContacts;
            for(i = 0; i < numContacts; ++i)
            {
                if(contacts.at(i)->penetration > max)
                {
                    max = contacts.at(i)->penetration;
                    index = i;
                }
            }
            // Check if the current object's index is equal to the number of contacts found
            if(index == numContacts)
                break;

            // TODO: AWAKE OPTIMIZATION

            // Resolve the penetration
            contacts.at(index)->applyPositionChange(linearChange, angularChange, max);

            // Since the penetration resolving may have changed the penetration of
            // other bodies, we update contacts.
            for(i = 0; i < numContacts; ++i)
            {
                // Check each body in the contact
                for(unsigned int b = 0; b < 2; ++b)
                {
                    if(contacts.at(i)->body[b] != nullptr)
                    {
                        for(unsigned int d = 0; d < 2; ++d)
                        {
                            if(contacts.at(i)->body[b] == contacts.at(index)->body[d])
                            {
                                // Calculate the new position of the relative contact point for each
                                // object, based on the linear and angular movements we applied
                                deltaPosition = linearChange[d] + 
                                    Athena::Vector3::cross(angularChange[d], contacts.at(i)->relativeContactPosition[b]);
                                
                                /**
                                 * The sign of the change is positive if we are
                                 * dealing with the second body in contact,
                                 * negative otherwise.
                                 * If they have moved apart (along the line of the
                                 * contact normal), then the penetration will be less;
                                 * if they overlapped the penetration will be increased
                                */
                                contacts.at(i)->penetration += deltaPosition.dot(contacts.at(i)->contactNormal) * (b != 0 ? 1 : -1);
                            }
                        }
                    }
                }
                this->positionIterationsUsed++;
            }
        }
    }

    void ContactResolver::adjustVelocities(std::vector<Contact*>& contacts, unsigned int numContacts, Athena::Scalar dt)
    {
        // This algorithm is very similar to the one used to resolve interpenetrations
        unsigned int i, index;
        Athena::Vector3 velocityChange[2], rotationChange[2];
        Athena::Vector3 deltaVelocity;

        velocityIterationsUsed = 0;
        while(velocityIterationsUsed < velocityIterations)
        {
            Athena::Scalar max = this->velocityEpsilon;
            index = numContacts;
            for(i = 0; i < numContacts; ++i)
            {
                if(contacts.at(i)->desiredDeltaVelocity > max)
                {
                    max = contacts.at(i)->desiredDeltaVelocity;
                    index = i;
                }
            }

            if(index == numContacts)
                break;

            contacts.at(index)->applyVelocityChange(velocityChange, rotationChange);

            for(i = 0; i < numContacts; ++i)
            {
                // Check each body in the contact
                for(unsigned int b = 0; b < 2; ++b)
                {
                    if(contacts.at(i)->body[b] != nullptr)
                    {
                        for(unsigned int d = 0; d < 2; ++d)
                        {
                            if(contacts.at(i)->body[b] == contacts.at(index)->body[d])
                            {
                                deltaVelocity = velocityChange[d] + 
                                    Athena::Vector3::cross(rotationChange[d], contacts.at(i)->relativeContactPosition[b]);
                                
                                contacts.at(i)->contactVelocity += contacts.at(i)->contactToWorldSpace.transformTranspose(deltaVelocity)
                                * (b != 0? -1 : 1);
                                contacts.at(i)->calculateDesiredDeltaVelocity(dt);
                            }
                        }
                    }
                }
            }
            velocityIterationsUsed++;
        }
    }

    void ContactResolver::resolveContacts(std::vector<Contact*>& contacts, unsigned int numContacts, Athena::Scalar dt)
    {
        if(numContacts == 0)
            return;
        
        if(velocityIterations <= 0)
            velocityIterations = numContacts * 4;

        if(positionIterations <= 0)
            positionIterations = numContacts * 4;

        if(!isValid())
            return;
        
        // Prepare the contacts for processing
        prepareContacts(contacts, numContacts, dt);

        // Resolve interpenetrations
        adjustPositions(contacts, numContacts, dt);

        // Resolve velocities
        adjustVelocities(contacts, numContacts, dt);
    }
}