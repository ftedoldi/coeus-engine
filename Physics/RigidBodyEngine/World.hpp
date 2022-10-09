#ifndef WORLD_HPP
#define WORLD_HPP
#include <RigidBody.hpp>
#include <ContactGenerator.hpp>
#include <ContactResolver.hpp>
#include <CollisionGenerator.hpp>
#include <GravityForce.hpp>
#include <vector>

namespace Khronos
{
    /**
     * The world class represents an indipendent simulation of physics.
     * It keep tracks of a set of rigid bodies and provides the means to
     * update them all.
    */
    class World
    {

        // Calls each of the registered contact generators to report
        // their contacts. Return the number of generated contacts    
        unsigned int generateContacts();

    public:

        //Holds a single rigid body in a vector of rigid bodies
        std::vector<RigidBody*> bodyList;

        // Holds the resolver for sets of contacts
        ContactResolver* resolver;

        // Holds a contact generators
        CollisionGenerator* collisionGenerator;

        // Holds an array of contacts
        std::vector<Contact*> contacts;

        // Holds an array of forces
        //std::vector<GravityForce*> forces;
        GravityForce* gForce;

        // Max number of contacts that can happen each frame
        unsigned int maxContacts;

        void clearContacts();

        World(unsigned int maxContacts, unsigned int iterations = 0);

        ~World();
        

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