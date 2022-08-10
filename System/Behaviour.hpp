#ifndef __BEHAVIOUR_H__
#define __BEHAVIOUR_H__

#include <Component.hpp>

namespace System
{
    class Behaviour : public Component {
        virtual void start() = 0;
        virtual void update() = 0;

        virtual void setOrderOfExecution(const short& newOrderOfExecution) = 0;

        virtual int getUniqueID() = 0;

        virtual std::string toString() = 0;
    };
}


#endif // __BEHAVIOUR_H__