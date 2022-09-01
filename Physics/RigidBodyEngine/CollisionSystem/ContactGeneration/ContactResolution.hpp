#ifndef CONTACTRESOLUTION_HPP
#define CONTACTRESOLUTION_HPP
#include <Contact.hpp>

namespace Khronos
{
    class ContactResolution
    {
    protected:

        /**
         * Sets up contacts ready for processing. This method makes sure
         * that their internal data is configured correctly and the 
         * correct set of bodies is made alive.
        */
        void prepareContacts(Contact* contactArray, unsigned int numContacts, Athena::Scalar dt);
    };
}

#endif