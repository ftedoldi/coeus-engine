#include "../ParticleSpringForce.hpp"

namespace Khronos
{
    ParticleSpringForce::ParticleSpringForce(Particle* other, Athena::Scalar springConstant, Athena::Scalar restLength):
        _other(other), _springConstant(springConstant), _restLength(restLength)
    {

    }

    void ParticleSpringForce::updateForce(Particle* particle, Athena::Scalar dt)
    {
        Athena::Vector3 force = particle->getPosition();

        force -= this->_other->getPosition();

        Athena::Scalar magnitude = force.magnitude();
        magnitude = std::abs(magnitude - this->_restLength);
        magnitude *= this->_springConstant;

        force.normalize();
        force *= -magnitude;
        particle->addForce(force);
    }
        
}