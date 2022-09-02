#ifndef __DEBUG_H__
#define __DEBUG_H__

#include <Console.hpp>

#include <string>

namespace System {
    class Debug {
        public:
            static EditorLayer::Console* mainConsole;
            
            static void Log(std::string message);
            static void LogError(std::string errorMessage);
            static void LogWarning(std::string warningMessage);
    };
}

#endif // __DEBUG_H__