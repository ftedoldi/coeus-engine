#ifndef __GUI_H__
#define __GUI_H__

#include <imgui.h>
#include <imgui_internal.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <Transform.hpp>

#include <filesystem>
#include <vector>

namespace EditorLayer
{
    class HierarchyWindow;
}

namespace System
{
    class Component;

    namespace Utils
    {
        class GUI
        {
            public:
                static std::vector<Component*> inspectorParameters;

                static void setEngineEditorStyle();
                static void setEngineRuntimeStyle();

                static void createDockableArea();

                static void displayFoldersAtPath(
                                                    std::filesystem::path sourceFolder, 
                                                    std::filesystem::path& currentDirectory, 
                                                    int index = 1
                                                );

                static void displayChildrenOfTransform  (
                                                            Odysseus::Transform* childrenTransform,
                                                            EditorLayer::HierarchyWindow* hierarchyWindow, 
                                                            int index = 1
                                                        );

                static void beginDragAndDroppableTransform(
                                                            EditorLayer::HierarchyWindow* hierarchyWindow,
                                                            Odysseus::Transform* transformToSetOnPreview
                                                          );

                static void selectTransform(EditorLayer::HierarchyWindow* hierarchyWindow, Odysseus::Transform*& transformToShow);

                static void loadInspectorParameters(Odysseus::Transform* transformToAnalyze);
        };
    } // namespace Utils
} // namespace System


#endif // __GUI_H__