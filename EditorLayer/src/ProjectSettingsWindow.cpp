#include "../ProjectSettingsWindow.hpp"

#include <imgui.h>
#include <imgui_internal.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <SceneManager.hpp>

namespace EditorLayer
{
    ProjectSettingsWindow::ProjectSettingsWindow()
    {

    }

    void ProjectSettingsWindow::draw()
    {
        ImGui::Begin("Project Settings");
        static char pathOfStandardScene[256] = {""};
        strcpy(pathOfStandardScene, Odysseus::SceneManager::activeScene->path.c_str());
        ImGui::InputText("Default Scene Path", pathOfStandardScene, sizeof(char) * 256);

        char sceneName[256] = {0};
        strcpy(sceneName, Odysseus::SceneManager::activeScene->name.c_str());
        ImGui::InputText("Scene Name", sceneName, sizeof(char) * 256);

        if (ImGui::IsItemActive())
            Odysseus::SceneManager::activeScene->name = sceneName;

        ImGui::End();
    }
} // namespace EditorLayer
