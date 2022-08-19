#ifndef PARTICLECONTACTSOLVER_HPP
#define PARTICLECONTACTSOLVER_HPP
#include <Scalar.hpp>
#include <ParticleContact.hpp>
#include <Vector3.hpp>

namespace Khronos
{
    class ParticleContactSolver
    {
    protected:
        //Number of iterations allowed
        unsigned int iterations;

        //Actual number of iterations used
        unsigned int iterationsUsed;
    
    public:

        //Create a new contact solver
        ParticleContactSolver(unsigned int iterations);

        //Sets the number of iterations
        void setIterations(unsigned int iterations);

        //Resolve a set of particle contacts for both interpenetration and velocity
        void resolveContact(ParticleContact* contactArray, unsigned int numContacts, Athena::Scalar dt);

    };
}

#endif