#include "../CollisionData.hpp"

namespace Khronos
{
    bool CollisionData::hasContactsLeft() const
    {
        return contactsLeft > 0;
    }

    void CollisionData::resetData(unsigned int maxContacts)
    {
        contactsLeft = maxContacts;
        contactCount = 0;
        contacts = contactArray;
    }

    void CollisionData::addContacts(unsigned int count)
    {
        contactsLeft -= count;
        contactCount += count;

        // Move the array forward by count positions
        contacts += count;
    }
}