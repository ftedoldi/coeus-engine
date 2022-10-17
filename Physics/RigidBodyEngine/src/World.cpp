#include "../World.hpp"

namespace Khronos
{

    World::World(unsigned int maxContacts, unsigned int iterations):
        bodyList(std::vector<RigidBody*>()),
        gForce(new GravityForce(Athena::Vector3(0.0, -9.81, 0.0))),
        collisionGenerator(new CollisionGenerator()),
        resolver(new ContactResolver(iterations)),
        maxContacts(maxContacts),
        contacts(std::vector<Contact*>(maxContacts))
    {
        for(unsigned int i = 0; i < this->contacts.size(); ++i)
        {
            this->contacts.at(i) = new Contact();
        }
    }

    World::~World()
    {
        delete this->gForce;
        delete this->resolver;
        delete this->collisionGenerator;
        for(unsigned int i = 0; i < this->contacts.size(); ++i)
        {
            delete this->contacts.at(i);
            this->contacts.at(i) = nullptr;
        }

        this->gForce = nullptr;
        this->resolver = nullptr;
        this->collisionGenerator = nullptr;

    }

    void World::startFrame()
    {
        for(auto body : this->bodyList)
        {
            //Removes all forces from the accumulator
            body->clearAccumulators();
            body->calculateDerivedData();
        }
    }

    unsigned int World::generateContacts()
    {
        clearContacts();
        unsigned int limit = maxContacts;
        unsigned int nextContact = 0;

        unsigned int contactsUsed = this->collisionGenerator->addContact(this->contacts, nextContact);
        limit -= contactsUsed;
        nextContact += contactsUsed;

        return maxContacts - limit;
    }

    void World::runPhysics(Athena::Scalar dt)
    {
        
        //dt = 0.0199999995529652;
        for(auto body : bodyList)
        {
            this->gForce->updateForce(body, dt);
        }

        for(auto body : this->bodyList)
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