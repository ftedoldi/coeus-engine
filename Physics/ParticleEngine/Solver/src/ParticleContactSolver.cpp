#include "../ParticleContactSolver.hpp"

namespace Khronos
{
    ParticleContactSolver::ParticleContactSolver(unsigned int iterations) : iterations(iterations)
    {

    }

    void ParticleContactSolver::setIterations(unsigned int iterations)
    {
        this->iterations = iterations;
    }

    void ParticleContactSolver::resolveContact(ParticleContact* contactArray, unsigned int numContacts, Athena::Scalar dt)
    {
        this->iterationsUsed = 0;
        while(iterationsUsed < iterations)
        {
            //Find the contact with lower separating velocity (largest closing velocity)
            Athena::Scalar max = 0;
            unsigned int maxIndex = numContacts;
            //For each contact, find the contact with largest closing velocity
            for(unsigned int i = 0; i < numContacts; ++i)
            {

                Athena::Scalar separateVelocity = contactArray[i].calculateSeparatingVelocity();
                if(separateVelocity < max && (separateVelocity < 0 || contactArray[i].penetration > 0))
                {
                    max = separateVelocity;
                    maxIndex = i;
                }
            }

            if(maxIndex == numContacts)
                break;
            
            //Resolve this contact
            contactArray[maxIndex].resolve(dt);

            //Update the interpenetration for all particles
            Athena::Vector3 *move = contactArray[maxIndex].particleMovement;
            for (unsigned int i = 0; i < numContacts; i++)
            {
                if (contactArray[i].particle[0] == contactArray[maxIndex].particle[0])
                {
                    contactArray[i].penetration -= move[0] * contactArray[i].contactNormal;
                }
                else if (contactArray[i].particle[0] == contactArray[maxIndex].particle[1])
                {
                    contactArray[i].penetration -= move[1] * contactArray[i].contactNormal;
                }
                if (contactArray[i].particle[1])
                {
                    if (contactArray[i].particle[1] == contactArray[maxIndex].particle[0])
                    {
                        contactArray[i].penetration += move[0] * contactArray[i].contactNormal;
                    }
                    else if (contactArray[i].particle[1] == contactArray[maxIndex].particle[1])
                    {
                        contactArray[i].penetration += move[1] * contactArray[i].contactNormal;
                    }
                }
            }
            iterationsUsed++;
        }
    }
}