#ifndef __SCENE_H__
#define __SCENE_H__

#include <SceneObject.hpp>
#include <Editor.hpp>

#include <vector>
#include <string>

namespace Odysseus
{
    class SceneObject;

    class Scene
    {
        private:
            std::vector<SceneObject*> _objectsInScene;

        public:
            System::Editor* sceneEditor;

            std::string name;
            const std::vector<SceneObject*>& objectsInScene;

            Scene();
            Scene(const std::string& name);

            bool deleteSceneObject(SceneObject* obj);
            bool addSceneObject(SceneObject* obj);

            void initialiseScene();
            void draw();
    };
} // namespace Odysseus


#endif // __SCENE_H__