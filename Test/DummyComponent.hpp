#ifndef __DUMMYCOMPONENT_H__
#define __DUMMYCOMPONENT_H__

#include <Camera.hpp>

#include <iostream>
#include <string>

namespace System {
    class Component;
}

class System::Component;

class DummyComponent : public System::Component {
    public:
        int var;
        
        DummyComponent();

        virtual void setOrderOfExecution(const short& newOrderOfExecution);

        virtual void start();
        virtual void update();

        virtual short getUniqueID();

        virtual std::string toString();

        virtual ~DummyComponent();
};

#endif // __DUMMYCOMPONENT_H__