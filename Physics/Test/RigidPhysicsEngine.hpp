#ifndef RIGIDPHYSICSENGINE_HPP
#define RIGIDPHYSICSENGINE_HPP
#include <World.hpp>
#include <Time.hpp>

namespace Khronos
{
    class RigidPhysicsEngine
    {
    public:

        int iterations = 0;

        int maxContacts = 100;

        double epsilon = 0.01;
        
        World* instance;

        RigidPhysicsEngine() : instance(new World(maxContacts, iterations))
        {
            instance->resolver->positionIterations = iterations;
            instance->resolver->velocityIterations = iterations;
            instance->resolver->positionEpsilon = epsilon;
            instance->resolver->velocityEpsilon = epsilon;
            instance->collisionGenerator->restitution = 0;
            instance->collisionGenerator->friction = 0;
        }

        ~RigidPhysicsEngine()
        {
            delete instance;
            instance = nullptr;
        }

        void update()
        {
            //instance->startFrame();
            //instance->runPhysics(System::Time::deltaTime);
        }
    };
}

#endif