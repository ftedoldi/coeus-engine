#ifndef PARTICLEFORCEREGISTRY_HPP
#define PARTICLEFORCEREGISTRY_HPP
#include <Particle.hpp>
#include <ParticleForceGenerator.hpp>
#include <vector>

namespace Khronos
{
    //Registry containing which force generator affects which particle
    class ParticleForceRegistry
    {  
        protected:
            //struct which keeps track of a single force generator and the particle it applies to
            struct ParticleForceRegistration
            {
                Particle *particle;
                ParticleForceGenerator *pfg;
            };

            typedef std::vector<ParticleForceRegistration> Registry;
            
            //Keeps the list of all registrations
            Registry registrations;

        public:
            //Adds a given force generator to the registry
            void add(Particle* particle, ParticleForceGenerator* pfg);

            //Remove a given force generator from the registry
            void remove(Particle* particle, ParticleForceGenerator* pfg);

            //Clear the registry
            void clear();

            //Call all the force generators in the registry to update their forces of the corresponding particles
            void updateForces(Athena::Scalar dt);

    };
}
#endif