#ifndef __DUMMYCOMPONENT_H__
#define __DUMMYCOMPONENT_H__

#include <Component.hpp>

#include <iostream>
#include <string>

class DummyComponent : public System::Component {
    public:
        int var;
        float asd;
        
        DummyComponent();

        virtual void setOrderOfExecution(const short& newOrderOfExecution);

        virtual void start();
        virtual void update();

        virtual int getUniqueID();

        virtual std::string toString();
    
        virtual ~DummyComponent();
};

#endif // __DUMMYCOMPONENT_H__