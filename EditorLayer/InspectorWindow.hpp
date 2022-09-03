#ifndef __INSPECTORWINDOW_H__
#define __INSPECTORWINDOW_H__

namespace EditorLayer
{
    struct InspectorWindowIcons
    {
        int pointLightTextureID;
        int spotLightTextureID;
        int directionalLightTextureID;
        int areaLightTextureID;
        int removeComponentTextureID;
    };

    class InspectorWindow
    {
        private:
            InspectorWindowIcons icons;

            void setupIcons();

        public:
            InspectorWindow();

            void draw();
    };
} // namespace EditorLayer


#endif // __INSPECTORWINDOW_H__