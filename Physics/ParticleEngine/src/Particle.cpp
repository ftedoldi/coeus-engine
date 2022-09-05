#include "../Particle.hpp"

namespace Khronos
{
    Particle::Particle()
    {

    }
    void Particle::setInverseMass(const Athena::Scalar inverseMass)
    {
        this->inverseMass = inverseMass;
    }

    Athena::Scalar Particle::getInverseMass() const
    {
        return this->inverseMass;
    }

    void Particle::setPosition(const Athena::Vector3& position)
    {
        this->position = position;
    }

    void Particle::setVelocity(const Athena::Vector3& setVelocity)
    {
        this->velocity = velocity;
    }

    void Particle::setAcceleration(const Athena::Vector3& acceleration)
    {
        this->acceleration = acceleration;
    }

    void Particle::setDamping(const Athena::Scalar damping)
    {
        this->damping = damping;
    }

    Athena::Vector3 Particle::getPosition() const
    {
        return this->position;
    }

    Athena::Vector3 Particle::getVelocity() const
    {
        return this->velocity;
    }

    Athena::Vector3 Particle::getAcceleration() const
    {
        return this->acceleration;
    }

    Athena::Scalar Particle::getDamping() const
    {
        return this->damping;
    }

    Athena::Scalar Particle::getMass() const
    {
        if(this->inverseMass == 0)
            return SCALAR_MAX;
        else
            return ((Athena::Scalar)1.0) / this->inverseMass;
    }

    void Particle::clearForceAccum()
    {
        this->forceAccum.coordinates.x = 0;
        this->forceAccum.coordinates.y = 0;
        this->forceAccum.coordinates.z = 0;
    }

    bool Particle::hasFiniteMass()
    {
        return this->inverseMass >= 0.0f;
    }
    
    void Particle::addForce(Athena::Vector3& force)
    {
        this->forceAccum += force;
    }

    void Particle::integrate(Athena::Scalar dt)
    {
        if(this->inverseMass <= 0.0)
            return;
        assert(dt > 0.0);

        //update position
        this->position += this->velocity * dt;

        //update acceleration
        Athena::Vector3 newAcceleration = this->acceleration;
        newAcceleration += this->forceAccum * this->inverseMass;

        //update velocity
        this->velocity += newAcceleration * dt;

        //update velocity based on damping factor
        //if we want to simulate a very high number
        //of objects, we remove the powf
        this->velocity *= std::powf(damping, dt);

        //We clear at each frame the forces acting on the particle
        clearForceAccum();
    }
}