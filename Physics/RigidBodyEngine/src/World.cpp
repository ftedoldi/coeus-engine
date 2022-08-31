#include "../World.hpp"

namespace Khronos
{
    void World::startFrame()
    {
        BodyRegistration* registration = this->head;
        while(registration)
        {
            //Removes alla forces from the accumulator
            registration->body->clearAccumulators();
            registration->body->calculateDerivedData();

            //go to the next registration
            registration = registration->next;
        }
    }

    void World::runPhysics(Athena::Scalar dt)
    {
        BodyRegistration* registration = this->head;
        while(registration)
        {
            registration->body->integrate(dt);

            registration = registration->next;
        }
    }
}