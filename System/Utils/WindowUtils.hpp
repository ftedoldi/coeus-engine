#ifndef __WINDOWUTILS_H__
#define __WINDOWUTILS_H__

#include <Window.hpp>

#include <string>

namespace System
{
    namespace Utils
    {
        class FileDialogs
        {
            public:
                // If an empty string is returned it means that the file dialog has been cancelled
                static std::string OpenFile(const char* filter, std::string title="File Dialog", std::string pathToLoadFromSourceDirectory="");
                // If an empty string is returned it means that the file dialog has been cancelled
                static std::string SaveFile(const char* filter, std::string title="File Dialog", std::string pathToLoadFromSourceDirectory="");
        };
    }
} // namespace System


#endif // __WINDOWUTILS_H__