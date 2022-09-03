#ifndef __CONSOLEWINDOW_H__
#define __CONSOLEWINDOW_H__

#include <Console.hpp>

namespace EditorLayer
{
    class ConsoleWindow
    {
        private:
            Console* console;

        public:
            ConsoleWindow();

            void draw();
    };
} // namespace EditorLayer

#endif // __CONSOLEWINDOW_H__