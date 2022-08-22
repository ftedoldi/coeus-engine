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

    bool Scene::deleteSceneObject(const int& i)
    {
        int size = _objectsInScene.size();

        deleteChildren(_objectsInScene[i]->transform);

        return _objectsInScene.size() < size;
    }

    // TODO: Implement me
    bool Scene::deleteSceneObject(SceneObject* obj)
    {
        for (int i = 0; i < _objectsInScene.size(); i++)
        {
            if (_objectsInScene[i]->ID == obj->ID)
            {
                deleteChildren(obj->transform);
                return true;
            }
        }

        return false;
    }

    void Scene::deleteChildren(Transform* t)
    {
        if (t->children.size() == 0)
        {
            int sceneObjIndex = getSceneObjectIndex(t->sceneObject);
            _objectsInScene.erase(_objectsInScene.begin() + sceneObjIndex);
            return;
        }

        for (int i = 0; i < t->children.size(); i++)
        {
            deleteChildren(t->children[i]);
        }

        int sceneObjIndex = getSceneObjectIndex(t->sceneObject);
        _objectsInScene.erase(_objectsInScene.begin() + sceneObjIndex);
        return;
    }

    int Scene::getSceneObjectIndex(SceneObject* obj)
    {
        for (int i = 0; i < _objectsInScene.size(); i++)
        {
            if (_objectsInScene[i]->ID == obj->ID)
            {
                return i;
            }
        }

        return -1;
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