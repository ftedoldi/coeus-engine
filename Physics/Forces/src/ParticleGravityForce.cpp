#include "../ParticleGravityForce.hpp"

namespace Khronos
{
    ParticleGravityForce::ParticleGravityForce(const Athena::Vector3& gravity)
    {
        this->_gravity = gravity;
    }

    void ParticleGravityForce::updateForce(Particle* particle, Athena::Scalar dt)
    {
        if(!particle->hasFiniteMass())
            return;
        
        particle->addForce(this->_gravity * particle->getMass());
    }
}