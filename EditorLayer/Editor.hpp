#ifndef __EDITOR_H__
#define __EDITOR_H__

namespace Odysseus
{
    class EditorCamera;
}

namespace EditorLayer
{
    class Editor
    {
        public:
            Odysseus::EditorCamera* editorCamera;

            Editor();
    };
} // namespace System


#endif // __EDITOR_H__