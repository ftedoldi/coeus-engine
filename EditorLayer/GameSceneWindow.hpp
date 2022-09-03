#ifndef __GAMESCENEWINDOW_H__
#define __GAMESCENEWINDOW_H__

#include <imgui.h>
#include <imgui_internal.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <ImGuizmo.h>

namespace EditorLayer
{
    class GameSceneWindow
    {
        public:
            GameSceneWindow();

            void draw();
    };
} // namespace EditorLayer


#endif // __GAMESCENEWINDOW_H__