#ifndef __SERIALIZABLEFIELD_H__
#define __SERIALIZABLEFIELD_H__

#include <Component.hpp>

#include <iostream>
#include <cstring>
#include <string>
#include <vector>
#include <unordered_map>

namespace System
{
    namespace Serializable
    {
        class SerializableField
        {
            private:
                static std::unordered_map<std::string, std::unordered_map<std::string, std::string>> serializableComponents;

            public:
                template<class S>
                static void serialize(System::Component className, S variableValue)
                {
                    auto varName = typeid(variableValue).name();
                    std::string stringVarName(varName);

                    switch (varName)
                    {
                    case stringVarName.find("class") == std::string::npos :
                        serializableComponents[className.toString()][varName] = variableValue;
                        break;
                    case stringVarName.find("class") != std::string::npos :
                        // TODO: Decompose the class into multiple components
                        break;
                    default:
                        std::cerr << "Could not read Component type" << std::endl;
                        break;
                    }
                }
        };
    } // namespace Serializable
    
} // namespace System


#endif // __SERIALIZABLEFIELD_H__