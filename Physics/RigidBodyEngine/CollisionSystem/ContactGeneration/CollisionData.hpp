#ifndef COLLISIONDATA_HPP
#define COLLISIONDATA_HPP
#include <Contact.hpp>

namespace Khronos
{
    class CollisionData
    {
    public:
        // Holds the first contact in the array
        Contact* contactArray;

        // Holds the contact array to write into
        Contact* contacts;

        // Holds the maximum number of contacts the array can take
        unsigned int contactsLeft;

        // Holds the number of contacts found so far
        unsigned int contactCount;

        // Holds the friction value to write into any collision
        Athena::Scalar friction;

        // Holds the restitution value to write into any collision
        Athena::Scalar restitution;

        bool hasContactsLeft() const;

        void resetData(unsigned int maxContacts);

        void addContacts(unsigned int count);
    };
}

#endif