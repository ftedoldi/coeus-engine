#ifndef PARTICLESPRINGFORCE_HPP
#define PARTICLESPRINGFORCE_HPP
#include <ParticleForceGenerator.hpp>
#include <Particle.hpp>
#include <Scalar.hpp>
#include <Vector3.hpp>

namespace Khronos
{
    class ParticleSpringForce : public ParticleForceGenerator
    {
        private:
            //Particle at the other end of the spring
            Particle* _other;

            //spring constant used in Hook's law
            Athena::Scalar _springConstant;

            //length of the spring when rested
            Athena::Scalar _restLength;

        public:
            ParticleSpringForce(Particle* other, Athena::Scalar springConstant, Athena::Scalar restLength);

            virtual void updateForce(Particle* particle, Athena::Scalar dt);

    };
}

#endif