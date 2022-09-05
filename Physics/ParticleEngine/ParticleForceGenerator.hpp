#ifndef PARTICLEFORCEGENERATOR_HPP
#define PARTICLEFORCEGENERATOR_HPP
#include <Particle.hpp>
#include <Scalar.hpp>

namespace Khronos
{
    class ParticleForceGenerator
    {
        public:
            //We calculate and update a specific force in this method implementation
            virtual void updateForce(Particle *particle, Athena::Scalar dt) = 0;
    };
}

#endif