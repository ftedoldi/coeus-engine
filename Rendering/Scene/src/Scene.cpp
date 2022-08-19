#include "../Scene.hpp"

namespace Odysseus
{

    Scene::Scene() : objectsInScene(_objectsInScene)
    {
        name = "Sample Scene";
    }

    Scene::Scene(const std::string& name) : objectsInScene(_objectsInScene)
    {
        this->name = name;
    }

    void Scene::initialiseScene()
    {
        for (int i = 0; i < _objectsInScene.size(); i++)
        {
            _objectsInScene[i]->transform->name = "SceneObj" + std::to_string(i); 
            for (int j = 0; j < _objectsInScene[i]->_container->components.size(); j++)
            {
                _objectsInScene[i]->_container->components[j]->start();
                glUseProgram(0);
            }
        }
    }

    // TODO: Implement me
    bool Scene::deleteSceneObject(SceneObject* obj)
    {
        return false;
    }

    bool Scene::addSceneObject(SceneObject* obj)
    {
        int size = _objectsInScene.size();

        _objectsInScene.push_back(obj);
    
        return size + 1 == _objectsInScene.size();
    }

    void Scene::draw()
    {
        for (int i = 0; i < _objectsInScene.size(); i++)
            for (int j = 0; j < _objectsInScene[i]->_container->components.size(); j++)
            {
                _objectsInScene[i]->_container->components[j]->update();
                glUseProgram(0);
            }
    }

}