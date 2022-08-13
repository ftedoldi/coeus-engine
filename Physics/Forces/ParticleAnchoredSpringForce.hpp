#ifndef PARTICLEANCHOREDSPRINGFORCE_HPP
#define PARTICLEANCHOREDSPRINGFORCE_HPP
#include <ParticleForceGenerator.hpp>
#include <Particle.hpp>
#include <Scalar.hpp>
#include <Vector3.hpp>

namespace Khronos
{
    class ParticleAnchoredSpringForce : public ParticleForceGenerator
    {
        private:
            //Position of the anchor point
            Athena::Vector3 _anchorPositon;

            //spring constant used in Hook's law
            Athena::Scalar _springConstant;

            //length of the spring when rested
            Athena::Scalar _restLength;

        public:
            ParticleAnchoredSpringForce(Athena::Vector3& anchorPos, Athena::Scalar springConstant, Athena::Scalar restLength);

            void setAnchorPos(Athena::Vector3& anchorPos);

            virtual void updateForce(Particle* particle, Athena::Scalar dt);

    };
}

#endif