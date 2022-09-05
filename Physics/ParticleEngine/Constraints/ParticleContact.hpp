#ifndef PARTICLECONTACT_HPP
#define PARTICLECONTACT_HPP
#include <Particle.hpp>
#include <Scalar.hpp>
#include <Vector3.hpp>

namespace Khronos
{
    class ParticleContactResolver;

    class ParticleContact
    {
        friend class ParticleContactResolver;

    public:
        /**
         * Particles that are involved in the contact.
         * If the particle have a contact with the scenery
         * the second particle is set to NULL.
        */
        Particle* particle[2];

        /**
         * Restitution coefficent at the contact between the particles.
         * It controls the speed at which the objects will separate
         * after colliding
        */
        Athena::Scalar restitution;

        /**
         * Direction in world space in which the two objects are colliding.
        */
        Athena::Vector3 contactNormal;

        /**
         * Depth of the penetration at the contact.
         * A negative depth represents two objects that have no interpenetration.
         * A zero depth represents two objects that are merely touching.
        */
        Athena::Scalar penetration;

        Athena::Vector3 particleMovement[2];

    //protected:
        /**
         * Resolves the contact for both velocity and interpenetration.
        */
        void resolve(Athena::Scalar dt);
        
        /**
         * Calculates the separating velocity at this contact.
        */
        Athena::Scalar calculateSeparatingVelocity() const;

    private:
        /**
         * Handles the impulse calculation for the collision.
         * Impulses are not calculated as we calculate forces
         * by accumulate all of them.
         * Each impulse will be applied one at the time after the collision.
        */
        void resolveVelocity(Athena::Scalar dt);

        /**
         * Handles the interpenetration resolution fot the contact.
        */
        void resolveInterpenetration(Athena::Scalar dt);

    };
}
#endif