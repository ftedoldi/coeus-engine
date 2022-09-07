#ifndef __HIERARCHYWINDOW_H__
#define __HIERARCHYWINDOW_H__

#include <Transform.hpp>

namespace System::Utils
{
    class GUI;
}

namespace EditorLayer
{
    class HierarchyWindow
    {
        friend class System::Utils::GUI;

        private:
            Odysseus::Transform* selectedItem;
            Odysseus::Transform* hoveredDraggingTransform;

            void drawPopupMenu();
            
        public:
            HierarchyWindow();

            void draw();
    };
} // namespace EditorLayer

#endif // __HIERARCHYWINDOW_H__