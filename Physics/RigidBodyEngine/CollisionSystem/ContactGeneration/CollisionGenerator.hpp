#ifndef COLLISIONGENERATOR_HPP
#define COLLISIONGENERATOR_HPP
#include <ContactGenerator.hpp>
#include <CollisionDetector.hpp>
//#include <vector>

namespace Khronos
{
    class CollisionGenerator : public ContactGenerator
    {
    public:

        Athena::Scalar friction;

        Athena::Scalar restitution;

        std::vector<CollisionPlane*> planes;
        std::vector<CollisionBox*> boxes;
        std::vector<CollisionSphere*> spheres;

        CollisionGenerator();

        ~CollisionGenerator();
        
        unsigned int addContact(std::vector<Contact*>& contacts, unsigned int next);

    private:

        void detectCollision(CollisionSphere* sphere, CollisionData* data);

        void detectCollision(CollisionBox* box, CollisionData* data);
    };
}

#endif