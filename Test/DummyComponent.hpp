#ifndef __DUMMYCOMPONENT_H__
#define __DUMMYCOMPONENT_H__

#include <SerializableClass.hpp>

#include <iostream>
#include <string>

namespace System
{
    class Component;
}

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

        SERIALIZABLE_CLASS();
};


SERIALIZABLE_FIELDS
{
    Serialize::SerializableClass::serialize<DummyComponent>().constructor<>()(rttr::policy::ctor::as_raw_ptr).property("var", &DummyComponent::var).property("asd", &DummyComponent::asd);
}

#endif // __DUMMYCOMPONENT_H__