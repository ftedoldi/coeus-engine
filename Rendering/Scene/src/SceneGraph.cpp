#include <SceneGraph.hpp>

namespace Odysseus {
    std::vector<SceneObject*> SceneGraph::objectsInScene;
    
    void SceneGraph::initializeScene()
    {
        System::Time::time = static_cast<Athena::Scalar>(glfwGetTime());
        System::Time::timeAtLastFrame = static_cast<Athena::Scalar>(glfwGetTime());
        System::Time::deltaTime = static_cast<Athena::Scalar>(glfwGetTime());

        for (int i = 0; i < objectsInScene.size(); i++)
            for (int j = 0; j < objectsInScene[i]->_container->components.size(); j++)
                objectsInScene[i]->_container->components[j]->start();
    }

    void SceneGraph::drawScene()
    {
        System::Time::time = static_cast<Athena::Scalar>(glfwGetTime());
        System::Time::deltaTime = System::Time::time - System::Time::timeAtLastFrame;
        System::Time::timeAtLastFrame = System::Time::time;

        for (int i = 0; i < objectsInScene.size(); i++)
            for (int j = 0; j < objectsInScene[i]->_container->components.size(); j++)
                objectsInScene[i]->_container->components[j]->update();
    }
}
