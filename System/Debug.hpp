#ifndef __DEBUG_H__
#define __DEBUG_H__

#include <Window.hpp>

#include <string>

namespace System {
    class Console{
        private:
            char                  InputBuf[256];
            ImVector<char*>       Items;
            ImVector<const char*> Commands;
            ImVector<char*>       History;
            int                   HistoryPos;    // -1: new line, 0..History.Size-1 browsing history.
            ImGuiTextFilter       Filter;
            bool                  AutoScroll;
            bool                  ScrollToBottom;

        public:

            Console();
            ~Console();

            static int   Stricmp(const char* s1, const char* s2);       
            static int   Strnicmp(const char* s1, const char* s2, int n);
            static char* Strdup(const char* s);                       
            static void  Strtrim(char* s);

            void AddLog(const char* fmt, ...) IM_FMTARGS(2);
            void ClearLog();

            void ExecCommand(const char* command_line);

            int TextEditCallback(ImGuiInputTextCallbackData* data);

            void Draw(const char* title, bool* p_open);                      
    };
    
    class Debug {
        friend Window;

        private:
            static Console* mainConsole;

        public:
            static void Log(std::string message);
            static void LogError(std::string errorMessage);
            static void LogWarning(std::string warningMessage);
    };
}

#endif // __DEBUG_H__