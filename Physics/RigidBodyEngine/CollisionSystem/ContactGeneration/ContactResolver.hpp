#ifndef CONTACTRESOLVER_HPP
#define CONTACTRESOLVER_HPP
#include <Contact.hpp>
#include <vector>

namespace Khronos
{
    class ContactResolver
    {
    public:
        /**
         * Holds the number of iterations to perform when resolving velocity.
        */
        unsigned int velocityIterations;

        /**
         * Holds the number of iterations to perform when resolving position.
        */
        unsigned int positionIterations;

        /**
         * To avoid instability, velocities smaller than this value are considered
         * to be zero.
         * If the value is too small, the simulation may be unstable.
         * If the value is too large, the bodies may interpenetrate visually.
        */
        Athena::Scalar velocityEpsilon;

        /**
         * To avoid instability, penetrations smaller than this value are considered
         * to be zero.
         * If the value is too small, the simulation may be unstable.
         * If the value is too large, the bodies may interpenetrate visually.
        */
        Athena::Scalar positionEpsilon;

        // Stores the number of velocity iterations used in the last resolve contact call
        unsigned int velocityIterationsUsed;

        // Stores the number of position iterations used in the last resolve contact call
        unsigned int positionIterationsUsed;

    private:
        // Check if the internal settings are valid
        bool validSettings;

    public:
        // Return true if the resolver has valid settings
        bool isValid();

        // Creates a new contact resolver with the given number of iterations and
        // optional epsilon values
        ContactResolver(unsigned int iterations,
                        Athena::Scalar velocityEpsilon = (Athena::Scalar)0.01,
                        Athena::Scalar positionEpsilon = (Athena::Scalar)0.01);
        
        // Creates a new contact resolver with the given number of iterations per
        // type and optional epsilon values
        ContactResolver(unsigned int velocityIterations,
                        unsigned int positionIterations,
                        Athena::Scalar velocityEpsilon = (Athena::Scalar)0.01,
                        Athena::Scalar positionEpsilon = (Athena::Scalar)0.01);
        
        // Set the number of iterations for each resolution stage
        void setIterations(unsigned int iterations);

        // Set the number of iterations for both resolution stage
        void setIterations(unsigned int velocityIterations, unsigned int positionIterations);

        // Set the tolerance value for velocity and position
        void setEpsilon(Athena::Scalar velocityEpsilon, Athena::Scalar positionEpsilon);

        void resolveContacts(std::vector<Contact*>& contacts, unsigned int numContacts, Athena::Scalar dt);

    protected:
        /**
         * Sets up contacts ready for processing. This method makes sure
         * that their internal data is configured correctly and the 
         * correct set of bodies is made alive.
        */
        void prepareContacts(std::vector<Contact*>& contacts, unsigned int numContacts, Athena::Scalar dt);

        /**
         * Resolves the positional issues by finding the worst penetration,
         * resolving it and updating the remaining contacts.
        */
        void adjustPositions(std::vector<Contact*>& contacts, unsigned int numContacts, Athena::Scalar dt);

        /**
         * Resolves the velocity issues
        */
       void adjustVelocities(std::vector<Contact*>& contacts, unsigned int numContacts, Athena::Scalar dt);

    };
}

#endif