#include "../MainMenuBar.hpp"

#include <Window.hpp>

#include <WindowUtils.hpp>
#include <Folder.hpp>

#include <Python.h>

namespace EditorLayer
{
    MainMenuBar::MainMenuBar(EditorLayer::StatusBar* mainStatusBar)
    {
        this->backgroundColor = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
        this->textColor = IM_COL32(0, 0, 0, 255);
        this->borderColor = IM_COL32(255, 255, 255, 255);

        this->_statusBar = mainStatusBar;

        Py_Initialize();
    }

    void MainMenuBar::setMainStatusBar(EditorLayer::StatusBar* mainStatusBar)
    {
        this->_statusBar = mainStatusBar;
    }

    void MainMenuBar::setBackgroundColor(const ImVec4& bgColor)
    {
        this->backgroundColor = bgColor;
    }

    void MainMenuBar::setTextColor(const ImU32& txtColor)
    {
        this->textColor = txtColor;
    }

    void MainMenuBar::setBorderColor(const ImU32& borderColor)
    {
        this->borderColor = borderColor;
    }

    void MainMenuBar::initializeShortcutActions()
    {
        if (glfwGetKey(System::Window::window, GLFW_KEY_LEFT_CONTROL) && glfwGetKey(System::Window::window, GLFW_KEY_S))
            this->saveSceneToSourceFile();
        
        if (glfwGetKey(System::Window::window, GLFW_KEY_LEFT_CONTROL) && glfwGetKey(System::Window::window, GLFW_KEY_D))
            this->saveSceneViaFileDialog();
        
        if (glfwGetKey(System::Window::window, GLFW_KEY_LEFT_CONTROL) && glfwGetKey(System::Window::window, GLFW_KEY_O))
            this->openSceneViaFileDialog();

        if (glfwGetKey(System::Window::window, GLFW_KEY_LEFT_CONTROL) && glfwGetKey(System::Window::window, GLFW_KEY_LEFT_SHIFT) && glfwGetKey(System::Window::window, GLFW_KEY_N))
            this->openNewSceneViaFileDialog();
        else if (glfwGetKey(System::Window::window, GLFW_KEY_LEFT_CONTROL) && glfwGetKey(System::Window::window, GLFW_KEY_N))
            this->openNewScene();
    }

    void MainMenuBar::saveSceneToSourceFile()
    {
        if (Odysseus::SceneManager::activeScene->isRuntimeScene)
            return;

        if (!Odysseus::SceneManager::activeScene->path.empty())
        {
            System::Serialize::Serializer serializer = System::Serialize::Serializer();
            serializer.serialize(Odysseus::SceneManager::activeScene->path);
            this->_statusBar->addStatus("Scene Saved!", EditorLayer::StatusBarTextColor::GREEN);
        }
        else
            this->saveSceneViaFileDialog();
    }

    void MainMenuBar::saveSceneViaFileDialog()
    {
        if (Odysseus::SceneManager::activeScene->isRuntimeScene)
            return;

        std::string filePath = System::Utils::FileDialogs::SaveFile("Coeus Scene (*.coeus)\0*.coeus\0", "Save Scene As", "\\Assets\\Scenes");

        if (filePath.empty())
            return;

        std::string filePathWithExtension = filePath + ".coeus";
        auto index = filePathWithExtension.find(System::Folder::getApplicationAbsolutePath());
        filePathWithExtension.replace(index, System::Folder::getApplicationAbsolutePath().length(), ".");
        Odysseus::SceneManager::activeScene->path = filePathWithExtension;

        System::Serialize::Serializer serializer = System::Serialize::Serializer();
        serializer.serialize(Odysseus::SceneManager::activeScene->path);

        this->_statusBar->addStatus("Scene Saved at new Path: " + filePathWithExtension, EditorLayer::StatusBarTextColor::GREEN);
    }

    void MainMenuBar::openSceneViaFileDialog()
    {
        std::string filePath = System::Utils::FileDialogs::OpenFile("Coeus Scene (*.coeus)\0*.coeus\0", "Open Scene At", "\\Assets\\Scenes");

        if (filePath.empty())
            return;

        auto index = filePath.find(System::Folder::getApplicationAbsolutePath());
        filePath.replace(index, System::Folder::getApplicationAbsolutePath().length(), ".");

        System::Serialize::Serializer serializer = System::Serialize::Serializer();
        serializer.deserialize(filePath);
        
        this->_statusBar->addStatus("Opening scene at Path: " + filePath, EditorLayer::StatusBarTextColor::GREEN);
    }

    void MainMenuBar::openNewSceneViaFileDialog()
    {
        std::string filePath = System::Utils::FileDialogs::SaveFile("Coeus Scene (*.coeus)\0*.coeus\0", "New Scene At", "\\Assets\\Scenes");

        if (filePath.empty())
            return;

        std::string filePathWithExtension = filePath + ".coeus";
        auto index = filePathWithExtension.find(System::Folder::getApplicationAbsolutePath());
        std::string scenePath = filePathWithExtension.replace(index, System::Folder::getApplicationAbsolutePath().length(), ".");

        std::string scenesPath = ".\\Assets\\Scenes";
        index = filePathWithExtension.find(scenesPath);
        std::string sceneName = filePathWithExtension.replace(index, scenesPath.length(), "");

        Odysseus::Scene* newScene = new Odysseus::Scene(scenePath, sceneName);
        Odysseus::SceneManager::addScene(newScene);
        Odysseus::SceneManager::activeScene = newScene;

        Odysseus::SceneManager::initializeActiveScene();

        System::Serialize::Serializer serializer = System::Serialize::Serializer();
        serializer.serialize(Odysseus::SceneManager::activeScene->path);

        this->_statusBar->addStatus("Opening new scene at Path: " + scenePath, EditorLayer::StatusBarTextColor::GREEN);
    }

    void MainMenuBar::openNewScene()
    {
        Odysseus::Scene* newScene = new Odysseus::Scene("New Scene");
        Odysseus::SceneManager::addScene(newScene);
        Odysseus::SceneManager::activeScene = newScene;

        Odysseus::SceneManager::initializeActiveScene();

        this->_statusBar->addStatus("Opening New Scene", EditorLayer::StatusBarTextColor::GREEN);
    }

    void MainMenuBar::runTextureEditor()
    {
        char filename[] = "TextureEditor.py";
        FILE* fp;

        fp = _Py_fopen(filename, "r");
        PyRun_SimpleFile(fp, filename);
    }

    void MainMenuBar::openTextureEditor()
    {
        this->runTextureEditor();
    }

    void MainMenuBar::draw()
    {
        this->initializeShortcutActions();

        ImGui::PushStyleColor(ImGuiCol_MenuBarBg, this->backgroundColor); // Menu bar background color
        ImGui::PushStyleColor(ImGuiCol_Text, this->textColor);
        ImGui::PushStyleColor(ImGuiCol_Border, this->borderColor);

        if (ImGui::BeginViewportSideBar("Main Menu Bar", ImGui::GetMainViewport(), ImGuiDir_Up, ImGui::GetFrameHeight(), ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_MenuBar))
        {
            if (ImGui::BeginMenuBar())
            {
                ImGui::PushStyleColor(ImGuiCol_PopupBg, ImVec4(1.0f, 1.0f, 1.0f, 1.0f)); // Menu bar background color
                if (ImGui::BeginMenu("File"))
                {
                    // The ... means that we are going to open a File Dialog

                    // TODO: Open a new empty scene
                    // TODO: Generate a Modal that asks if we really wanna leave the current scene without saving
                    if (ImGui::MenuItem("New", "CTRL+N"))
                        this->openNewScene();
                    if (ImGui::MenuItem("New Scene...", "CTRL+SHIFT+N"))
                        this->openNewSceneViaFileDialog();
                    if (ImGui::MenuItem("Open Scene...", "CTRL+O"))
                        this->openSceneViaFileDialog();

                    ImGui::Separator();

                    if (ImGui::MenuItem("Save", "CTRL+S"))
                        this->saveSceneToSourceFile();
                    if (ImGui::MenuItem("Save As...", "CTRL+D"))
                        this->saveSceneViaFileDialog();

                    // if (ImGui::MenuItem("Flag: NoSplit",                "", (dockspace_flags & ImGuiDockNodeFlags_NoSplit) != 0))                 
                    //     dockspace_flags ^= ImGuiDockNodeFlags_NoSplit;

                    // if (ImGui::MenuItem("Flag: NoResize",               "", (dockspace_flags & ImGuiDockNodeFlags_NoResize) != 0))                
                    //     dockspace_flags ^= ImGuiDockNodeFlags_NoResize;

                    // if (ImGui::MenuItem("Flag: NoDockingInCentralNode", "", (dockspace_flags & ImGuiDockNodeFlags_NoDockingInCentralNode) != 0))  
                    //     dockspace_flags ^= ImGuiDockNodeFlags_NoDockingInCentralNode;

                    // if (ImGui::MenuItem("Flag: AutoHideTabBar",         "", (dockspace_flags & ImGuiDockNodeFlags_AutoHideTabBar) != 0))          
                    //     dockspace_flags ^= ImGuiDockNodeFlags_AutoHideTabBar;

                    // if (ImGui::MenuItem("Flag: PassthruCentralNode",    "", (dockspace_flags & ImGuiDockNodeFlags_PassthruCentralNode) != 0, opt_fullscreen)) 
                    //     dockspace_flags ^= ImGuiDockNodeFlags_PassthruCentralNode;

                    // ImGui::Separator();

                    // if (ImGui::MenuItem("Close", NULL, false, p_open != NULL))
                    //     *p_open = false;
                    ImGui::EndMenu();
                }
                if (ImGui::BeginMenu("Edit"))
                {
                    if (ImGui::MenuItem("Texture Editor..."))
                        this->openTextureEditor();
                    ImGui::EndMenu();
                }
                ImGui::PopStyleColor();
                ImGui::EndMenuBar();
            }
            
            ImGui::End();
        }

        ImGui::PopStyleColor(3);
    }

    MainMenuBar::~MainMenuBar()
    {
        Py_Finalize();
    }
} // namespace EditorLayer
