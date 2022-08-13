#ifndef PARTICLE_HPP
#define PARTICLE_HPP
#include <Vector3.hpp>
#include <assert.h>
#define SCALAR_MAX FLT_MAX

namespace Khronos
{
    class Particle
    {
        protected:
        //position and velocity are going to altered only by the integrator
        //not manually
        Athena::Vector3 position;
        Athena::Vector3 velocity;
        
        //used to set the acceleration due to gravity or
        //any other constant acceleration
        Athena::Vector3 acceleration;

        /*
        approximation of drag force
        damping = 0 velocity will be reduced to nothing
        damping = 1 object keeps all its velocity (equivalent of no damping)
        */
        Athena::Scalar damping;

        /**
         * we compute inverse of the mass since
         * if we want to compute the acceleration
         * we can easily represent objects with
         * infinite mass instead of zero mass
         * 
        */
        Athena::Scalar inverseMass;

        /**
         * Holds the accumulated force to be applied at the next
         * simulation iteration only. This value is zeroed at each
         * integration step.
         * We consider all forces action on an object as a single force
         * obtained as sum of all forces.
         */
        Athena::Vector3 forceAccum;


        public:
            Particle();
            void setInverseMass(const Athena::Scalar inverseMass);
            void setPosition(const Athena::Vector3& position);
            void setVelocity(const Athena::Vector3& setVelocity);
            void setAcceleration(const Athena::Vector3& acceleration);
            void setDamping(const Athena::Scalar damping);
            //void setForce(const Athena::Vector3& force);

            Athena::Vector3 getPosition() const;
            Athena::Vector3 getVelocity()const;
            Athena::Vector3 getAcceleration() const;
            Athena::Scalar getDamping() const;
            
            Athena::Scalar getInverseMass() const;
            Athena::Scalar getMass() const;

            void integrate(Athena::Scalar dt);
            void clearForceAccum();

            //return true if the mass is not infinite
            bool hasFiniteMass();
            
            //Forces added with this method are usually
            //used only for few frames
            void addForce(Athena::Vector3& force);

    };
}

#endif