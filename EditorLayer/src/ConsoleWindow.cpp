#include "../ConsoleWindow.hpp"

#include <Debug.hpp>

namespace EditorLayer
{
    
    ConsoleWindow::ConsoleWindow()
    {
        this->console = new Console();
        System::Debug::mainConsole = this->console;
    }
    
    void ConsoleWindow::draw()
    {
        static bool isOpen = true;
        console->Draw("Console", &isOpen);
    }

} // namespace EditorLayer
