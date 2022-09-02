#ifndef __SCENE_H__
#define __SCENE_H__

#include <SceneObject.hpp>
#include <Editor.hpp>

#include <vector>
#include <string>

namespace Odysseus
{
    class SceneObject;

    enum SceneState
    {
        EDITOR,
        RUNNING,
        PAUSED,
        STOPPED
    };

    class Scene
    {
        private:
            std::vector<SceneObject*> _objectsInScene;

            void deleteChildren(Transform* t);
            int getSceneObjectIndex(SceneObject* obj);

            void initialiseEditorScene();
            void initialiseRuntimeScene();

            void updateEditorScene();
            void updateRuntimeScene();

        public:
            bool isRuntimeScene;

            SceneState status;

            EditorLayer::Editor* sceneEditor;

            std::string path;

            std::string name;
            const std::vector<SceneObject*>& objectsInScene;

            Scene();
            Scene(const std::string& name);
            Scene(const std::string& path, const std::string& name);
            Scene(Scene* sceneToCopy, SceneState state, bool runtimeScene=false);

            bool addSceneObject(SceneObject* obj);

            bool deleteSceneObject(const int& i);
            bool deleteSceneObject(SceneObject* obj);

            void initialiseScene();
            void draw();
    };
} // namespace Odysseus


#endif // __SCENE_H__