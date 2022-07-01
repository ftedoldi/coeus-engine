#include <SceneGraph.hpp>

namespace Odysseus {
    std::vector<SceneObject*> SceneGraph::objectsInScene;
    
    void SceneGraph::initializeScene()
    {
        for (int i = 0; i < objectsInScene.size(); i++)
            for (int j = 0; j < objectsInScene[i]->_container->components.size(); j++)
                objectsInScene[i]->_container->components[j]->start();
    }

    void SceneGraph::drawScene()
    {
        for (int i = 0; i < objectsInScene.size(); i++)
            for (int j = 0; j < objectsInScene[i]->_container->components.size(); j++)
                objectsInScene[i]->_container->components[j]->update();
    }
}
