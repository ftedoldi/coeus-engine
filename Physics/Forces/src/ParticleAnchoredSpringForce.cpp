#include "../ParticleAnchoredSpringForce.hpp"

namespace Khronos
{
    ParticleAnchoredSpringForce::ParticleAnchoredSpringForce(Athena::Vector3& anchorPos, Athena::Scalar springConstant, Athena::Scalar restLength):
        _anchorPositon(anchorPos), _springConstant(springConstant), _restLength(restLength)
    {

    }
    void ParticleAnchoredSpringForce::setAnchorPos(Athena::Vector3& anchorPos)
    {
        this->_anchorPositon = anchorPos;
    }

    void ParticleAnchoredSpringForce::updateForce(Particle* particle, Athena::Scalar dt)
    {
        Athena::Vector3 force = particle->getPosition();

        force -= this->_anchorPositon;

        Athena::Scalar magnitude = force.magnitude();
        magnitude = std::abs(magnitude - this->_restLength);
        magnitude *= this->_springConstant;

        force.normalize();
        force *= -magnitude;
        particle->addForce(force);
    }

}