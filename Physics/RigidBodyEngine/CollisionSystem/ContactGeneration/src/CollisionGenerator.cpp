#include "../CollisionGenerator.hpp"

namespace Khronos
{
    CollisionGenerator::CollisionGenerator()
    {
    }


    unsigned int CollisionGenerator::addContact(std::vector<Contact*>& contacts, unsigned int next)
    {
        auto data = new CollisionData();
        data->contactArray = contacts;
        data->resetData(next);
        data->friction = this->friction;
        data->restitution = this->restitution;

        for(auto sphere : this->spheres)
        {
            sphere->calculateInternals();
            if(!data->hasContactsLeft())
                break;
            detectCollision(sphere, data);
        }

        for(auto box : this->boxes)
        {
            box->calculateInternals();
            if(!data->hasContactsLeft())
                break;
            detectCollision(box, data);
        }
        int contactCount = data->contactCount;
        delete data;
        data = nullptr;
        return contactCount;
    }

    void CollisionGenerator::detectCollision(CollisionSphere* sphere, CollisionData* data)
    {
        for(auto plane : this->planes)
        {
            CollisionDetector::sphereAndHalfSpace(sphere, plane, data);
        }

        for(auto sphere2 : this->spheres)
        {
            if(sphere == sphere2)
                continue;
            if(!data->hasContactsLeft())
                break;
            
            CollisionDetector::sphereAndSphere(sphere, sphere2, data);
        }

        for(auto box : this->boxes)
        {
            if(!data->hasContactsLeft());

            CollisionDetector::boxAndSphere(box, sphere, data);
        }
    }

    void CollisionGenerator::detectCollision(CollisionBox* box, CollisionData* data)
    {
        for(auto plane : this->planes)
        {
            CollisionDetector::boxAndHalfSpace(box, plane, data);
        }

        for(auto sphere : this->spheres)
        {
            if(!data->hasContactsLeft())
                return;
            
            CollisionDetector::boxAndSphere(box, sphere, data);
        }

        // TODO:implement box-box collision detection
        /*for(auto box : *this->boxes)
        {
            if(!data->hasContactsLeft());

            CollisionDetector::boxAndSphere(box, sphere, data);
        }*/
    }
}