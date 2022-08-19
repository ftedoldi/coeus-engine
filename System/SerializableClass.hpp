#ifndef __SERIALIZABLEFIELD_H__
#define __SERIALIZABLEFIELD_H__

#include <Component.hpp>

#include <rttr/registration>

#include <iostream>
#include <string>
#include <unordered_map>

#define SERIALIZABLE_FIELDS         RTTR_REGISTRATION
// Pass as argument all the parent classes
#define SERIALIZABLE_CLASS(...)     RTTR_ENABLE(System::Component, TYPE_LIST(__VA_ARGS__))
                                 

typedef rttr::registration serialize;

namespace System
{
    namespace Serializable
    {
        class SerializableClass
        {
            public:
                template<class T>
                static rttr::registration::class_<T> serialize()
                {
                    return rttr::registration::class_<T>(T().toString());
                }
        };
    } // namespace Serializable
    
} // namespace System

namespace Serialize = System::Serializable;

#endif // __SERIALIZABLEFIELD_H__