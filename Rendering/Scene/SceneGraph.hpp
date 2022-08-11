#ifndef __SCENEGRAPH_H__
#define __SCENEGRAPH_H__

// #include <Window.hpp>

#include <Scalar.hpp>
#include <SceneObject.hpp>
#include <Time.hpp>

#include <vector>
#include <string>

namespace Odysseus {
    class SceneObject;
    
    // TODO: Refactor this with a non static class -> Entity Component System
    class SceneGraph {
        public:
            static std::string name;
            static std::vector<SceneObject*> objectsInScene;

            static void initializeScene();
            static void drawScene();
    };
}

#endif // __SCENEGRAPH_H__