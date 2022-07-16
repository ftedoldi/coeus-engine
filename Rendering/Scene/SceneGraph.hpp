#ifndef __SCENEGRAPH_H__
#define __SCENEGRAPH_H__

#include <Window.hpp>

#include <Scalar.hpp>
#include <SceneObject.hpp>
#include <Time.hpp>

#include <vector>

namespace Odysseus {
    class SceneObject;
    
    class SceneGraph {
        public:
            static std::vector<SceneObject*> objectsInScene;

            static void initializeScene();
            static void drawScene();
    };
}

#endif // __SCENEGRAPH_H__