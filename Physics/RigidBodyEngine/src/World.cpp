#include "../World.hpp"

namespace Khronos
{

    World::World(unsigned int maxContacts, unsigned int iterations):
        bodies(std::vector<RigidBody*>()),
        gForce(new GravityForce(Athena::Vector3(0.0, -9.81, 0.0))),
        collisionGenerator(new CollisionGenerator()),
        resolver(new ContactResolver(iterations)),
        maxContacts(maxContacts),
        contacts(std::vector<Contact*>(maxContacts))
    {
        //auto vec = std::vector<Contact*>(maxContacts);
        //this->contacts = vec;
    }

    World::~World()
    {
        delete this->resolver;
        delete this->collisionGenerator;
    }

    void World::startFrame()
    {
        for(auto body : this->bodies)
        {
            //Removes alla forces from the accumulator
            body->clearAccumulators();
            body->calculateDerivedData();
        }
    }

    /*void World::applyForces(Athena::Scalar dt)
    {
        for(auto body : bodies)
        {
            this->gForce->updateForce(body, dt);
        }
    }*/

    unsigned int World::generateContacts()
    {
        clearContacts();
        unsigned int limit = maxContacts;
        unsigned int nextContact = 0;

        //for(auto gen : this->collisionGenerators)
        //{
            unsigned int used = this->collisionGenerator->addContact(this->bodies, this->contacts, nextContact);
            limit -= used;
            nextContact += used;

            //if(limit <= 0)
            //    break;
        //}
        return maxContacts - limit;
    }

    void World::runPhysics(Athena::Scalar dt)
    {

        for(auto body : bodies)
        {
            this->gForce->updateForce(body, dt);
        }

        for(auto body : this->bodies)
        {
            body->integrate(dt);
        }

        unsigned int usedContacts = generateContacts();

        if(usedContacts > 0)
        {
            this->resolver->resolveContacts(contacts, usedContacts, dt);
        }
    }

    void World::clearContacts()
    {
        for(auto contact : contacts)
        {
            contact->clear();
        }
    }
}