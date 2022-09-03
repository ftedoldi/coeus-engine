#ifndef __INSPECTORSCENEWINDOW_H__
#define __INSPECTORSCENEWINDOW_H__

#include <imgui.h>
#include <imgui_internal.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <ImGuizmo.h>

#include <StatusBar.hpp>

namespace EditorLayer
{
    class InspectorSceneWindow
    {
        private:
            StatusBar* _mainStatusBar;
            ImGuizmo::OPERATION& gizmoOperation;

            void createObjectsGUIZMO();

        public:
            InspectorSceneWindow(StatusBar*& mainStatusBar, ImGuizmo::OPERATION& gizmoOp);

            void draw();
    };
} // namespace EditorLayer


#endif // __INSPECTORSCENEWINDOW_H__