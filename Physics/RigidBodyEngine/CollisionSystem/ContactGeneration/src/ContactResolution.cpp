#include "../ContactResolution.hpp"

namespace Khronos
{
    void ContactResolution::prepareContacts(Contact* contacts, unsigned int numContacts, Athena::Scalar dt)
    {
        // Generate contact velocity and axis informations
        Contact* lastContact = contacts + numContacts;
        for(Contact* contact = contacts; contact < lastContact; ++contact)
        {
            // Calculate the internal contact data
            contact->calculateInternals(dt);
        }
    }
}