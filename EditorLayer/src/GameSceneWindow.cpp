#include "../GameSceneWindow.hpp"

#include <Window.hpp>

namespace EditorLayer
{
    GameSceneWindow::GameSceneWindow()
    {

    }

    void GameSceneWindow::draw()
    {
        ImGui::Begin("Game");
        ImGui::BeginChild("Game Render");
        ImDrawList *dList = ImGui::GetWindowDrawList();
        dList->AddCallback((ImDrawCallback)&System::Window::sceneFrameBuffer->framebufferShaderCallback, System::Window::sceneFrameBuffer);
        ImGuizmo::SetRect(
            ImGui::GetWindowPos().x, ImGui::GetWindowPos().y,
            System::Window::sceneFrameBuffer->frameBufferSize.width,
            System::Window::sceneFrameBuffer->frameBufferSize.height);
        ImVec2 size = ImGui::GetWindowSize();

        // System::Window::frameBufferSize.width = size.x;
        // System::Window::frameBufferSize.height = size.y;

        // System::Window::sceneFrameBuffer->blit();

#pragma warning(push)
#pragma warning(disable : 4312)
        ImGui::Image((ImTextureID)System::Window::sceneFrameBuffer->texturesID[0], size, ImVec2(0, 1), ImVec2(1, 0));
#pragma warning(pop)

        dList->AddCallback(ImDrawCallback_ResetRenderState, nullptr);
        ImGui::EndChild();
        ImGui::End();
    }
} // namespace EditorLayer
