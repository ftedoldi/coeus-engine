#include <SceneManager.hpp>

#include <Editor.hpp>

namespace Odysseus {
    std::vector<Scene*> SceneManager::_loadedScenes;
    Scene* SceneManager::activeScene;

    // TODO: Print error message if add gone wrong
    void SceneManager::addScene(Scene* sceneToLoad)
    {
        if (sceneToLoad != nullptr)
            _loadedScenes.push_back(sceneToLoad);
    }

    bool SceneManager::setActiveScene(std::string name)
    {
        for (auto scene : _loadedScenes)
            if (scene->name == name)
            {
                activeScene = scene;
                activeScene->initialiseScene();
                return true;
            }
        
        return false;
    }

    bool SceneManager::setActiveScene(int i)
    {
        for (int index = 0; index < _loadedScenes.size(); index++)
            if (index == i)
            {
                activeScene = _loadedScenes[i];
                activeScene->initialiseScene();
                return true;
            }
        
        return false;
    }

    void SceneManager::initializeActiveScene()
    {
        System::Time::time = static_cast<Athena::Scalar>(glfwGetTime());
        System::Time::timeAtLastFrame = static_cast<Athena::Scalar>(glfwGetTime());
        System::Time::deltaTime = static_cast<Athena::Scalar>(glfwGetTime());

        System::Editor* editor = new System::Editor();
        activeScene->sceneEditor = editor;

        activeScene->initialiseScene();
    }

    void SceneManager::drawActiveScene()
    {
        System::Time::time = static_cast<Athena::Scalar>(glfwGetTime());
        System::Time::deltaTime = System::Time::time - System::Time::timeAtLastFrame;
        System::Time::timeAtLastFrame = System::Time::time;

        activeScene->draw();
    }
}
