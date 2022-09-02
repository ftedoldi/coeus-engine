#include "../StatusBar.hpp"

#include <imgui.h>
#include <imgui_internal.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

namespace EditorLayer {

    StatusBar::StatusBar()
    {
        errorStatus.statusText = " ";
        errorStatus.statusTextColor = StatusBarTextColor::RED;
    }

    bool StatusBar::isEmpty()
    {
        return this->statusQueue.size() == 0;
    }

    void StatusBar::addStatus(const std::string& text, StatusBarTextColor textColor)
    {
        StatusBarInfo status;

        status.statusText = text;
        status.statusTextColor = textColor;

        if (getLastStatus().statusText != text)
            this->statusQueue.push_back(status);
    }

    StatusBarInfo StatusBar::popStatus()
    {
        StatusBarInfo lastStatus = this->statusQueue[this->statusQueue.size() - 1];
        this->statusQueue.pop_back();

        return lastStatus;
    }

    StatusBarInfo StatusBar::popStatus(const short& i)
    {
        StatusBarInfo poppedStatus = this->statusQueue[this->statusQueue.size() - 1];
        this->statusQueue.erase(this->statusQueue.begin() + i);

        return poppedStatus;
    }

    StatusBarInfo StatusBar::getLastStatus()
    {
        if (this->statusQueue.size() > 0)
            return this->statusQueue[this->statusQueue.size() - 1];
        
        return errorStatus;
    }

    void StatusBar::draw()
    {
        ImGui::PushStyleColor(ImGuiCol_Border, IM_COL32(255,255,255,0));

        if (ImGui::BeginViewportSideBar("Status Bar", ImGui::GetMainViewport(), ImGuiDir_Down, ImGui::GetFrameHeight(), ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_MenuBar))
        {
            if (ImGui::BeginMenuBar()) {
                    static EditorLayer::StatusBarInfo statusToDisplay = this->errorStatus;

                    if (statusToDisplay.statusTextColor == EditorLayer::StatusBarTextColor::RED)
                        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1, 0, 0, 1));
                    else if (statusToDisplay.statusTextColor == EditorLayer::StatusBarTextColor::GREEN)
                        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0, 1, 0, 1));
                    else if (statusToDisplay.statusTextColor == EditorLayer::StatusBarTextColor::WHITE)
                        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1, 1, 1, 1));
                    else if (statusToDisplay.statusTextColor == EditorLayer::StatusBarTextColor::YELLOW)
                        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1, 1, 0, 1));
                        
                    if (this->getLastStatus().statusText != this->errorStatus.statusText)
                        statusToDisplay = this->popStatus();

                    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + ImGui::GetColumnWidth() - ImGui::CalcTextSize(statusToDisplay.statusText.c_str()).x 
                        - ImGui::GetScrollX() - 2 * ImGui::GetStyle().ItemSpacing.x);

                    ImGui::Text("%s", statusToDisplay.statusText.c_str());
                    ImGui::PopStyleColor();
                ImGui::EndMenuBar();
            }
            ImGui::End();
        }

        ImGui::PopStyleColor();
    }

}