#include "../WindowUtils.hpp"

#pragma warning(push)
#pragma warning(disable : 4005) 

#include <Windows.h>
#include <Commdlg.h>

// #include <Window.hpp>
#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>

#include <Folder.hpp>

namespace System::Utils
{

    std::string FileDialogs::OpenFile(const char* filter, std::string title, std::string pathToLoadFromSourceDirectory)
    {
        OPENFILENAMEA ofn;
        CHAR szFile[260] = { 0 };

        ZeroMemory(&ofn, sizeof(OPENFILENAME));

        ofn.lStructSize = sizeof(OPENFILENAME);
        ofn.hwndOwner = glfwGetWin32Window(System::Window::window);
        ofn.lpstrFile = szFile;
        ofn.nMaxFile = sizeof(szFile);
        ofn.lpstrFilter = filter;
        ofn.nFilterIndex = 1;
        ofn.lpstrTitle = title.c_str();
        ofn.lpstrInitialDir = (Folder::getApplicationAbsolutePath() + pathToLoadFromSourceDirectory).c_str();
        ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;

        if (GetOpenFileName(&ofn) == TRUE)
            return ofn.lpstrFile;

        return std::string();
    }

    std::string FileDialogs::SaveFile(const char* filter, std::string title, std::string pathToLoadFromSourceDirectory)
    {
        OPENFILENAMEA ofn;
        CHAR szFile[260] = { 0 };

        ZeroMemory(&ofn, sizeof(OPENFILENAME));

        ofn.lStructSize = sizeof(OPENFILENAME);
        ofn.hwndOwner = glfwGetWin32Window(System::Window::window);
        ofn.lpstrFile = szFile;
        ofn.nMaxFile = sizeof(szFile);
        ofn.lpstrFilter = filter;
        ofn.nFilterIndex = 1;
        ofn.lpstrTitle = title.c_str();
        ofn.lpstrInitialDir = (Folder::getApplicationAbsolutePath() + pathToLoadFromSourceDirectory).c_str();
        ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;

        if (GetSaveFileName(&ofn) == TRUE)
            return ofn.lpstrFile;

        return std::string();
    }

} // namespace System::Utils

#pragma warning(pop)
