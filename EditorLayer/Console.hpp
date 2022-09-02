#ifndef __CONSOLE_H__
#define __CONSOLE_H__

#include <imgui.h>
#include <imgui_internal.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <string>

namespace EditorLayer
{
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
} // namespace Editor

#endif // __CONSOLE_H__