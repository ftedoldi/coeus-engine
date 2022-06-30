#ifndef __SCENEGRAPH_H__
#define __SCENEGRAPH_H__

#include "SceneObject.hpp"

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