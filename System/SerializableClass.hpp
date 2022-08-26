#ifndef __SERIALIZABLEFIELD_H__
#define __SERIALIZABLEFIELD_H__

#include <Component.hpp>

#include <rttr/registration>

#include <iostream>
#include <string>
#include <vector>

// Specify the parent class/classes here
#define SERIALIZABLE_CLASS(...) RTTR_ENABLE(__VA_ARGS__)

// Use this macro to specify wich class you wish to register
#define SERIALIZE_CLASS RTTR_REGISTRATION

// Use this macro in order to get a variable name
#define NAMEOF(variable) ((decltype(&variable))nullptr, #variable)

namespace System
{
    class SerializableClass
    {
        public:
            template<class T>
            static rttr::registration::class_<T> registerClass(std::string name)
            {
                return rttr::registration::class_<T>(name).constructor<>()(rttr::policy::ctor::as_raw_ptr);
            }
    };
} // namespace System

#endif // __SERIALIZABLEFIELD_H__