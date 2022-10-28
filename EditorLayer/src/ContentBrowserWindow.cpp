#include "../ContentBrowserWindow.hpp"

#include <imgui.h>
#include <imgui_internal.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <Model.hpp>
#include <Texture2D.hpp>

#include <Folder.hpp>
#include <UUID.hpp>

#include <stb/stb_image.h>

#include <filesystem>

namespace EditorLayer
{

    ContentBrowserWindow::ContentBrowserWindow()
    {
        this->initializeIcons();
    }

    void ContentBrowserWindow::initializeIcons()
    {
        stbi_set_flip_vertically_on_load(false);

        icons.leftArrowTextureID = Odysseus::Texture2D::loadTextureFromFile(
                                                                                (System::Folder::getFolderPath("Icons").string() + "/leftArrow.png").c_str(),
                                                                                true
                                                                            ).ID;
        icons.reloadTextureID = Odysseus::Texture2D::loadTextureFromFile(
                                                                            (System::Folder::getFolderPath("Icons").string() + "/rotate.png").c_str(),
                                                                            true
                                                                        ).ID;
        icons.folderTextureID = Odysseus::Texture2D::loadTextureFromFile(
                                                                            (System::Folder::getFolderPath("Icons").string() + "/folder.png").c_str(),
                                                                            true
                                                                        ).ID;
        icons.documentTextureID = Odysseus::Texture2D::loadTextureFromFile(
                                                                                (System::Folder::getFolderPath("Icons").string() + "/document.png").c_str(),
                                                                                true
                                                                            ).ID;
        icons.sceneTextureID = Odysseus::Texture2D::loadTextureFromFile(
                                                                            (System::Folder::getFolderPath("Icons").string() + "/scene.png").c_str(),
                                                                            true
                                                                        ).ID;
        icons.modelTextureID = Odysseus::Texture2D::loadTextureFromFile(
                                                                            (System::Folder::getFolderPath("Icons").string() + "/model.png").c_str(),
                                                                            true
                                                                        ).ID;

        stbi_set_flip_vertically_on_load(true);
    }

    // TODO: Refactor this
    void ContentBrowserWindow::draw()
    {
        static int actionIndex = 0;
        static std::vector<std::filesystem::path> actions = { System::Folder::assetDirectory };

        static ImGuiTextFilter filter;

        ImGui::Begin("Project", NULL, ImGuiWindowFlags_NoScrollbar);
        ImGui::Columns(2);

        static bool isFirstOpening = true;
        if (isFirstOpening)
        {
            ImGui::SetColumnWidth(0, ImGui::GetContentRegionAvail().x / 1.8);
            isFirstOpening = false;
        }

        if (ImGui::CollapsingHeader(System::Folder::assetDirectory.filename().string().c_str()))
        {
            if (ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left))
            {
                System::Folder::currentDirectory = System::Folder::assetDirectory;
            }
            System::Utils::GUI::displayFoldersAtPath(System::Folder::assetDirectory, System::Folder::currentDirectory);
        }

        ImGui::NextColumn();

        ImGui::BeginChild("Inner", {0, 0}, false, ImGuiWindowFlags_NoScrollbar);
        static float iconScale = 24;

        ImGui::PushStyleColor(ImGuiCol_Button, {0, 0, 0, 0});
        if (System::Folder::currentDirectory.string() != System::Folder::assetDirectory.string())
        {
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.38f, 0.38f, 0.38f, 0.50f));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.67f, 0.67f, 0.67f, 0.39f));
        }
        else
        {
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, {0, 0, 0, 0});
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, {0, 0, 0, 0});
        }

#pragma warning(push)
#pragma warning(disable : 4312)
        ImGui::ImageButtonEx(
            System::UUID(),
            (ImTextureID)icons.leftArrowTextureID,
            {iconScale, iconScale},
            {0, 0},
            {1, 1},
            {0, 0},
            {0, 0, 0, 0},
            {1, 1, 1, 1});
#pragma warning(pop)

        if (System::Folder::currentDirectory.string() != System::Folder::assetDirectory.string())
            if (ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left))
            {
                System::Folder::currentDirectory = System::Folder::currentDirectory.parent_path();

                actionIndex -= 1;
            }

        ImGui::SameLine();

#pragma warning(push)
#pragma warning(disable : 4312)
        ImGui::ImageButtonEx(
            System::UUID(),
            (ImTextureID)icons.leftArrowTextureID,
            {iconScale, iconScale},
            {1, 1},
            {0, 0},
            {0, 0},
            {0, 0, 0, 0},
            {1, 1, 1, 1});
#pragma warning(pop)

        if (actions.size() > 1 && (actionIndex + 1) < actions.size())
            if (ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left))
                System::Folder::currentDirectory = actions[++actionIndex];

        ImGui::SameLine();

#pragma warning(push)
#pragma warning(disable : 4312)
        ImGui::ImageButtonEx(
            System::UUID(),
            (ImTextureID)icons.reloadTextureID,
            {iconScale, iconScale},
            {1, 1},
            {0, 0},
            {0, 0},
            {0, 0, 0, 0},
            {1, 1, 1, 1});
#pragma warning(pop)

        if (ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left))
            std::fill_n(filter.InputBuf, 256, 0);

        ImGui::PopStyleColor(3);

        ImGui::SameLine();

        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.4f, 0.4f, 0.4f, 1.00f));
        float filterWidth = ImGui::GetContentRegionAvail().x / 4 + 40;
        filter.Draw(" ", filterWidth);
        static std::string filterContent(filter.InputBuf);
        filterContent = filter.InputBuf;
        std::for_each(filterContent.begin(), filterContent.end(), [](char &c)
                      { c = ::tolower(c); });
        ImGui::PopStyleColor();

        ImGui::SameLine();

        ImGui::Text("Assets");

        ImGui::Separator();

        // TODO: move this settings inside a file in order to let the user set his custom values
        static float padding = 22;
        static float thumbnailSize = ImGui::GetContentRegionAvail().x / 9;

        float cellSize = thumbnailSize + padding;
        float panelWidth = ImGui::GetContentRegionAvail().x;

        int columnCount = (int)(panelWidth / cellSize) < 1 ? 1 : (int)(panelWidth / cellSize);
        ImGui::Columns(columnCount, 0, false);

        auto index = 0;

        for (auto &directory : std::filesystem::directory_iterator(System::Folder::currentDirectory))
        {
            auto &path = directory.path();
            auto relativePath = std::filesystem::relative(path, System::Folder::currentDirectory);
            std::string filenameString = relativePath.filename().string();
            std::string lowercaseFilenameString(filenameString);
            std::for_each(lowercaseFilenameString.begin(), lowercaseFilenameString.end(), [](char &c)
                          { c = ::tolower(c); });

            ImGui::PushStyleColor(ImGuiCol_Button, {0, 0, 0, 0});
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.38f, 0.38f, 0.38f, 0.50f));
            if (directory.is_directory() && (lowercaseFilenameString.find(filterContent) != std::string::npos))
            {

#pragma warning(push)
#pragma warning(disable : 4312)
                ImGui::ImageButtonEx(
                    System::UUID(),
                    (ImTextureID)icons.folderTextureID,
                    {thumbnailSize, thumbnailSize},
                    {0, 0},
                    {1, 1},
                    {10, 10},
                    {0, 0, 0, 0},
                    {1, 1, 1, 1});
#pragma warning(pop)

                if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left))
                {
                    System::Folder::currentDirectory = path;

                    if (actionIndex == 0 && actions.size() > 1)
                    {
                        actions.clear();
                        actions.push_back(System::Folder::assetDirectory);
                    }

                    actions.push_back(path);
                    actionIndex += 1;
                }
            }
            else if (lowercaseFilenameString.find(filterContent) != std::string::npos)
            {

                if (lowercaseFilenameString.find(".coeus") != std::string::npos)
                {
#pragma warning(push)
#pragma warning(disable : 4312)
                    ImGui::ImageButton(
                        (ImTextureID)icons.sceneTextureID,
                        {thumbnailSize, thumbnailSize},
                        {0, 0},
                        {1, 1},
                        10,
                        {0, 0, 0, 0},
                        {1, 1, 1, 1});
#pragma warning(pop)

                    ImGuiWindow *window = ImGui::GetCurrentWindow();
                    ImGui::ButtonBehavior(ImRect(ImGui::GetItemRectMin(), ImGui::GetItemRectMax()), ImGui::GetItemID(), NULL, NULL,
                                          ImGuiButtonFlags_MouseButtonMiddle | ImGuiButtonFlags_MouseButtonRight | ImGuiButtonFlags_MouseButtonLeft);

                    if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left))
                    {
                        System::Serialize::Serializer serializer = System::Serialize::Serializer();

                        System::Utils::GUI::inspectorParameters.clear();
                        Odysseus::SceneManager::activeScene->sceneEditor->selectedTransform = nullptr;

                        serializer.deserialize(path.string());
                    }

                    static std::string currentPath("");
                    if (ImGui::IsItemHovered())
                        currentPath = path.string();

                    if (ImGui::BeginDragDropSource())
                    {
                        ImGui::SetDragDropPayload("SCENE_FILE", currentPath.c_str(), (strlen(currentPath.c_str()) + 1) * sizeof(char));
                        ImGui::EndDragDropSource();
                    }
                    else
                        currentPath = "";
                }
                else if (lowercaseFilenameString.find(".obj")  != std::string::npos ||
                         lowercaseFilenameString.find(".fbx")  != std::string::npos ||
                         lowercaseFilenameString.find(".gltf") != std::string::npos)
                {
#pragma warning(push)
#pragma warning(disable : 4312)
                    ImGui::ImageButton(
                        (ImTextureID)icons.modelTextureID,
                        {thumbnailSize, thumbnailSize},
                        {0, 0},
                        {1, 1},
                        10,
                        {0, 0, 0, 0},
                        {1, 1, 1, 1});
#pragma warning(pop)

                    static std::string currentPath("");
                    if (ImGui::IsItemHovered())
                        currentPath = path.string();

                    if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left))
                    {
                        Odysseus::Shader *modelShader = new Odysseus::Shader(".\\Shader\\phongShader.vert", ".\\Shader\\phongShader.frag");
                        //Odysseus::Model myModel(path.string(), modelShader, false);
                        Odysseus::SceneManager::initializeActiveScene();
                    }

                    if (ImGui::BeginDragDropSource())
                    {
                        ImGui::SetDragDropPayload("MODEL_FILE", currentPath.c_str(), (strlen(currentPath.c_str()) + 1) * sizeof(char));
                        ImGui::EndDragDropSource();
                    }
                    else
                        currentPath = "";
                }
                else
                {
#pragma warning(push)
#pragma warning(disable : 4312)
                    ImGui::ImageButtonEx(
                        System::UUID(),
                        (ImTextureID)icons.documentTextureID,
                        {thumbnailSize, thumbnailSize},
                        {0, 0},
                        {1, 1},
                        {10, 10},
                        {0, 0, 0, 0},
                        {1, 1, 1, 1});
#pragma warning(pop)

                    if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left))
                        system(path.string().c_str());
                }
            }
            ImGui::PopStyleColor(2);

            if (lowercaseFilenameString.find(filterContent) != std::string::npos)
            {
                ImGui::TextWrapped(filenameString.c_str());
                ImGui::NextColumn();
            }
        }

        ImGui::Columns(1);
        ImGui::EndChild();

        ImGui::Columns(1);
        ImGui::End();
    }
} // namespace EditorLayer
