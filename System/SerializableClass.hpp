#ifndef __SERIALIZABLEFIELD_H__
#define __SERIALIZABLEFIELD_H__

#include <Component.hpp>

#include <iostream>
#include <string>
#include <unordered_map>

namespace System
{
    class SerializableClass
    {
        public:
            static std::unordered_map<std::string, size_t> m;
    };
} // namespace System

#endif // __SERIALIZABLEFIELD_H__