#ifndef PARTICLEGRAVITYFORCE_HPP
#define PARTICLEGRAVITYFORCE_HPP
#include <ParticleForceGenerator.hpp>
#include <Vector3.hpp>
#include <Scalar.hpp>
#include <Particle.hpp>

namespace Khronos
{
    class ParticleGravityForce : public ParticleForceGenerator
    {
        private:
        Athena::Vector3 _gravity;

        public:
            ParticleGravityForce(const Athena::Vector3& gravity);
            virtual void updateForce(Particle* particle, Athena::Scalar dt);
    };
}

#endif