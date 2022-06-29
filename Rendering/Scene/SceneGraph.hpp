#ifndef __SCENEGRAPH_H__
#define __SCENEGRAPH_H__

#include "SceneObject.hpp"

#include <vector>

namespace Odysseus {
    class SceneGraph {
        private:
            static std::vector<SceneObject*> objectsInScene;

        public:
            static void drawScene();
    };
}

#endif // __SCENEGRAPH_H__