#include "../ParticleContact.hpp"

namespace Khronos
{
    void ParticleContact::resolve(Athena::Scalar dt)
    {
        this->resolveVelocity(dt);
        this->resolveInterpenetration(dt);
    }

    Athena::Scalar ParticleContact::calculateSeparatingVelocity() const
    {
        Athena::Vector3 relativeVelocity = particle[0]->getVelocity();
        /**
         * If the second particle exist (we have contact between two particles)
         * we compute the total relative velocity
         * else we consider only the first particle relative velocity
        */
        if(particle[1] != nullptr)
            relativeVelocity -= particle[1]->getVelocity();
        
        //dot product
        return relativeVelocity * this->contactNormal;
    }

    void ParticleContact::resolveVelocity(Athena::Scalar dt)
    {
        //Theory start at pag 126 to 131
        Athena::Scalar separatingVelocity = this->calculateSeparatingVelocity();

        if(separatingVelocity > 0)
        {
            //No impulse required since if the separating velocity
            //is greater than 0, the contact is separating or is
            //stationary
            return;
        }

        //Calculate the individual velocity of each object
        Athena::Scalar newSeparatingVelocity = -separatingVelocity * this->restitution;

        //Calculate the separating velocity due to the acceleration only in the direction
        //of the contact normal.
        Athena::Vector3 accCausedVelocity = particle[0]->getAcceleration();

        //If the second particle exists, we compute the new acceleration used to calculate
        //the separating velocity
        if(particle[1] != nullptr)
            accCausedVelocity -= particle[1]->getAcceleration();
        
        Athena::Scalar accCausedSeparateVelocity = accCausedVelocity * this->contactNormal * dt;

        if(accCausedSeparateVelocity < 0)
        {
            newSeparatingVelocity += this->restitution * accCausedSeparateVelocity;
            if(newSeparatingVelocity < 0)
                newSeparatingVelocity = 0;
        }

        //Calculate the delta because its needed to calculate the strenght of the impulse
        Athena::Scalar deltaSeparatingVelocity = newSeparatingVelocity - separatingVelocity;

        Athena::Scalar totalInverseMass = particle[0]->getInverseMass();

        if(particle[1] != nullptr)
            totalInverseMass += particle[1]->getInverseMass();
        
        //if all particles have infinite mass, impulses have no effect
        if(totalInverseMass <= 0)
            return;

        //Calculate the impulse to apply
        Athena::Scalar impulse = deltaSeparatingVelocity / totalInverseMass;

        //Calculate the direction of the impulse
        Athena::Vector3 impulsePerInvMass = this->contactNormal * impulse;

        //Apply the impulses
        particle[0]->setVelocity(particle[0]->getVelocity() + impulsePerInvMass * particle[0]->getInverseMass());

        //if the second particle exists, it will go to the opposite direction as the first particle
        if(particle[1] != nullptr)
        {
            particle[1]->setVelocity(particle[1]->getVelocity() + impulsePerInvMass * -particle[1]->getInverseMass());
        }
    }

    void ParticleContact::resolveInterpenetration(Athena::Scalar dt)
    {
        /**
         * If we have no penetration, we resolve nothing
        */
        if(this->penetration <= 0)
            return;
        
        //Calculate the total inverse mass
        Athena::Scalar totalInverseMass = particle[0]->getInverseMass();
        if(particle[1] != nullptr)
            totalInverseMass += particle[1]->getInverseMass();
        
        if(totalInverseMass <= 0)
            return;
        
        //Find the amount of penetration resolution per unit of inverse mass
        Athena::Vector3 movePerInvMass = this->contactNormal * (totalInverseMass * this->penetration);

        //Calculate the movement amout based on the particle mass
        this->particleMovement[0] = movePerInvMass * this->particle[0]->getInverseMass();

        if(this->particle[1] != nullptr)
            particleMovement[1] = movePerInvMass * -this->particle[1]->getInverseMass();
        else
        {
            particleMovement[1].coordinates.x = 0;
            particleMovement[1].coordinates.y = 0;
            particleMovement[1].coordinates.z = 0;
        }

        //Apply the resolutiuon
        particle[0]->setPosition(particle[0]->getPosition() + particleMovement[0]);

        if(particle[1] != nullptr)
            particle[1]->setPosition(particle[1]->getPosition() + particleMovement[1]);
            
    }
}