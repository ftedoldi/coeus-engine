#ifndef PARTICLEDRAGFORCE_HPP
#define PARTICLEDRAGFORCE_HPP
#include <ParticleForceGenerator.hpp>
#include <Vector3.hpp>
#include <Scalar.hpp>
#include <Particle.hpp>

namespace Khronos
{
    class ParticleDragForce : public ParticleForceGenerator
    {
        private:
        //Velocity drag coefficent
        Athena::Scalar _k1;
        //Velocity squared drag coefficent
        Athena::Scalar _k2;

        public:
            ParticleDragForce(const Athena::Scalar k1, const Athena::Scalar k2);
            virtual void updateForce(Particle* particle, Athena::Scalar dt);
    };
}

#endif