#ifndef __SCENEGRAPH_H__
#define __SCENEGRAPH_H__

#include "SceneObject.hpp"

#include <vector>

namespace Odysseus {
    class SceneGraph {
        public:
            SceneGraph();

            void drawScene();

            ~SceneGraph();

            friend SceneObject;
    };
}

#endif // __SCENEGRAPH_H__