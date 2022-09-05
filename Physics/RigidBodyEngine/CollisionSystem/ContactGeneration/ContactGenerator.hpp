#ifndef CONTACTGENERATOR_HPP
#define CONTACTGENERATOR_HPP
#include <Contact.hpp>

namespace Khronos
{
    class ContactGenerator
    {
    public:
    
        virtual unsigned int addContact(std::vector<RigidBody*>bodies, std::vector<Contact*> contacts, unsigned int next) = 0;

    };
}

#endif