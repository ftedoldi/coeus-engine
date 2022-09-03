#ifndef __CONTENTBROWSERWINDOW_H__
#define __CONTENTBROWSERWINDOW_H__

namespace EditorLayer
{
    struct ContentBrowserIcons
    {
        int leftArrowTextureID;
        int reloadTextureID;
        int folderTextureID;
        int documentTextureID;
        int sceneTextureID;
        int modelTextureID;
    };
    
    class ContentBrowserWindow
    {
        private:
            ContentBrowserIcons icons;
            
            void initializeIcons();

        public:
            ContentBrowserWindow();

            void draw();
    };
} // namespace EditorLayer

#endif // __CONTENTBROWSERWINDOW_H__