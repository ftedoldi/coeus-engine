#include "../ParticleDragForce.hpp"

namespace Khronos
{

    ParticleDragForce::ParticleDragForce(const Athena::Scalar k1, const Athena::Scalar k2) : _k1(k1), _k2(k2)
    {

    }

    void ParticleDragForce::updateForce(Particle* particle, Athena::Scalar dt)
    {
        //calculating drag force based on object properties
        Athena::Vector3 force = particle->getVelocity();

        Athena::Scalar forceMagnitude = force.magnitude();
        Athena::Scalar dragCoeff = this->_k1 * forceMagnitude + this->_k2 * (forceMagnitude * forceMagnitude);

        //calculate total drag force
        force.normalize();
        force *= -dragCoeff;
        particle->addForce(force);
    }
}