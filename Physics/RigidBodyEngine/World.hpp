#ifndef WORLD_HPP
#define WORLD_HPP
#include <RigidBody.hpp>

namespace Khronos
{
    /**
     * The world class represents an indipendent simulation of physics.
     * It keep tracks of a set of rigid bodies and provides the means to
     * update them all.
    */
    class World
    {
        //Holds a single rigid body in a linked list of rigid bodies
        struct BodyRegistration
        {
            RigidBody* body;
            BodyRegistration* next;
        };

        //Holds the head of the list of registred bodies
        BodyRegistration* head;
    public:

        /**
         * Initializes the world for a simulation frame.
         * After calling this method, the rigid bodies can have their forces
         * and torques for this frame added.
        */
        void startFrame();

        //processes all the physics in the world
        void runPhysics(Athena::Scalar dt);

    };
}

#endif