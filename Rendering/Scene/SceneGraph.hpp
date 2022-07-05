#ifndef __SCENEGRAPH_H__
#define __SCENEGRAPH_H__

#include <GLFW/glfw3.h>

#include <Scalar.hpp>
#include <SceneObject.hpp>
#include <Time.hpp>

#include <vector>

namespace Odysseus {
    class SceneGraph {
        public:
            static std::vector<SceneObject*> objectsInScene;

            static void initializeScene();
            static void drawScene();
    };
}

#endif // __SCENEGRAPH_H__