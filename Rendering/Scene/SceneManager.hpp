#ifndef __SCENEGRAPH_H__
#define __SCENEGRAPH_H__

#include <Scalar.hpp>
#include <SceneObject.hpp>
#include <Scene.hpp>
#include <Time.hpp>

#include <vector>
#include <string>

namespace Odysseus {
    class SceneObject;
    class Scene;
    
    // TODO: Refactor this with a non static class -> Entity Component System
    class SceneManager {
        public:
            static std::vector<Scene*> _loadedScenes;
            static Scene* activeScene;

            static void addScene(Scene* sceneToLoad);

            static bool setActiveScene(std::string name);
            static bool setActiveScene(int i);

            static void initializeActiveScene();
            static void drawActiveScene();
    };
}

#endif // __SCENEGRAPH_H__