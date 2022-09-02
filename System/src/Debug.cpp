#include "../Debug.hpp"

namespace System {
    
    EditorLayer::Console* Debug::mainConsole;

    void Debug::Log(std::string message)
    {
        mainConsole->AddLog(message.c_str());
    }

    void Debug::LogError(std::string errorMessage)
    {
        mainConsole->AddLog(("[Error]: " + errorMessage).c_str());
    }

    void Debug::LogWarning(std::string warningMessage)
    {
        mainConsole->AddLog(("[Warning]: " + warningMessage).c_str());
    }

}