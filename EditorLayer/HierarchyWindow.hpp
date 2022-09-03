#ifndef __HIERARCHYWINDOW_H__
#define __HIERARCHYWINDOW_H__

#include <Transform.hpp>

namespace EditorLayer
{
    class HierarchyWindow
    {
        private:
            Odysseus::Transform *selectedItem;

            void drawPopupMenu();
            
        public:
            HierarchyWindow();

            void draw();
    };
} // namespace EditorLayer

#endif // __HIERARCHYWINDOW_H__