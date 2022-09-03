#ifndef __EDITOR_H__
#define __EDITOR_H__

namespace Odysseus
{
    class EditorCamera;
    class Transform;
}

namespace System
{
    class Dockspace;
}

namespace EditorLayer
{
    class Editor
    {
        private:
            System::Dockspace* dockedSpace;

            void setupEditorCamera();

            void handleMousePicking();

        public:
            Odysseus::EditorCamera* editorCamera;
            Odysseus::Transform* selectedTransform;

            Editor();

            void onEditorUpdate();
    };
} // namespace System


#endif // __EDITOR_H__