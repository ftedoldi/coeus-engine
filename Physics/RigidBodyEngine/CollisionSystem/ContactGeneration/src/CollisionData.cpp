#include "../CollisionData.hpp"

namespace Khronos
{
    bool CollisionData::hasContactsLeft() const
    {
        return contactsLeft > 0;
    }

    void CollisionData::resetData(unsigned int start)
    {
        contactsLeft = contactArray.size() - start;
        contactCount = start;
    }

    void CollisionData::addContacts(unsigned int count)
    {
        contactsLeft -= count;
        contactCount += count;
    }

    Contact* CollisionData::getContact()
    {
        if(!hasContactsLeft())
           throw std::exception("No contacts left");
        
        auto contact = contactArray.at(contactCount);
        addContacts(1);
        return contact;
    }
}