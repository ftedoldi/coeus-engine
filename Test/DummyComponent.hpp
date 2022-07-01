#ifndef __DUMMYCOMPONENT_H__
#define __DUMMYCOMPONENT_H__

#include <Component.hpp>

#include <iostream>
#include <string>

class DummyComponent : public Odysseus::Component {
    public:
        int var;
        
        DummyComponent();

        virtual void setOrderOfExecution(const short& newOrderOfExecution);

        virtual void start();
        virtual void update();

        virtual std::string getUniqueID();

        virtual std::string toString();

        virtual ~DummyComponent();
};

#endif // __DUMMYCOMPONENT_H__