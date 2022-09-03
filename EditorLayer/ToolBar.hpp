#ifndef __TOOLBAR_H__
#define __TOOLBAR_H__

#include <imgui.h>
#include <imgui_internal.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <ImGuizmo.h>

#include <StatusBar.hpp>

namespace EditorLayer
{
    struct ToolBarIcons
    {
        int translateTextureID;
        int scaleTextureID;
        int rotateTextureID;
        int playTextureID;
        int pauseTextureID;
        int stopTextureID;
    };

    class ToolBar
    {
        private:
            ToolBarIcons icons;
            StatusBar* _mainStatusBar;
            ImGuizmo::OPERATION& gizmoOperation;

            void initializeIcons();

            void initializeShortcuts();

            void translateObject();
            void rotateObject();
            void scaleObject();

            void resumeScene();
            void pauseScene();
            void playScene();
            void stopScene();

        public:
            ToolBar(StatusBar*& mainStatusBar, ImGuizmo::OPERATION& currentGizmoOperation);

            void setStatusBar(StatusBar*& mainStatusBar);

            void draw();
    };
} // namespace EditorLayer


#endif // __TOOLBAR_H__