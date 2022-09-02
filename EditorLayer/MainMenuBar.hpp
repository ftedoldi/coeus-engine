#ifndef __MAIN_MENUBAR_H__
#define __MAIN_MENUBAR_H__

#include <imgui.h>
#include <imgui_internal.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <StatusBar.hpp>

namespace EditorLayer
{
    class MainMenuBar
    {
        private:
            EditorLayer::StatusBar* _statusBar;

            ImVec4 backgroundColor;
            ImU32 textColor;
            ImU32 borderColor;

            void initializeShortcutActions();

            void saveSceneToSourceFile();
            void saveSceneViaFileDialog();
            void openSceneViaFileDialog();
            void openNewSceneViaFileDialog();
            void openNewScene();

        public:
            MainMenuBar();
            MainMenuBar(EditorLayer::StatusBar* mainStatusBar);

            void setBackgroundColor(const ImVec4& bgColor);
            void setTextColor(const ImU32& txtColor);
            void setBorderColor(const ImU32& borderColor);

            void setMainStatusBar(EditorLayer::StatusBar* mainStatusBar);

            void draw();
    };
} // namespace Editor


#endif // __MAIN_MENUBAR_H__