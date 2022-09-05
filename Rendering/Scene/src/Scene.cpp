#include "../Scene.hpp"

#include <Window.hpp>

#include <Component.hpp>

#include <iostream>
#include <fstream>

namespace Odysseus
{

    Scene::Scene() : objectsInScene(_objectsInScene)
    {
        name = "EmptyScene";
        this->path = "";

        this->isRuntimeScene = false;
        
        this->status = SceneState::EDITOR;
    }

    Scene::Scene(const std::string& name) : objectsInScene(_objectsInScene)
    {
        this->name = name;
        this->path = "";
        
        this->isRuntimeScene = false;
        
        this->status = SceneState::EDITOR;
    }

    Scene::Scene(const std::string& path, const std::string& name) : objectsInScene(_objectsInScene)
    {
        this->name = name;
        this->path = path;
        
        this->isRuntimeScene = false;

        this->status = SceneState::EDITOR;
    }

    Scene::Scene(Scene* sceneToCopy, SceneState state, bool runtimeScene) : objectsInScene(_objectsInScene)
    {
        this->name = sceneToCopy->name;
        this->path = sceneToCopy->path;

        this->_objectsInScene = sceneToCopy->_objectsInScene;
        this->sceneEditor = sceneToCopy->sceneEditor;
        
        this->isRuntimeScene = runtimeScene;

        this->status = state;
    }

    void Scene::initialiseEditorScene()
    {
        for (int i = 0; i < _objectsInScene.size(); i++)
        {
            for (int j = 0; j < _objectsInScene[i]->_container->components.size(); j++)
            {
                _objectsInScene[i]->_container->components[j]->start();
                glUseProgram(0);
            }
        }
    }

    // TODO: Implement me
    void Scene::initialiseRuntimeScene()
    {

    }

    void Scene::updateEditorScene()
    {
        for (int i = 0; i < _objectsInScene.size(); i++)
            for (int j = 0; j < _objectsInScene[i]->_container->components.size(); j++)
            {
                _objectsInScene[i]->_container->components[j]->update();
                glUseProgram(0);
            }
    }

    // TODO: Implement me
    void Scene::updateRuntimeScene()
    {

    }

    void Scene::initialiseScene()
    {
        this->initialiseEditorScene();

        if (this->isRuntimeScene)
            this->initialiseRuntimeScene();    
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
            Odysseus::SceneObject* obj = _objectsInScene[sceneObjIndex];
            _objectsInScene.erase(_objectsInScene.begin() + sceneObjIndex);
            delete obj;
            return;
        }

        for (int i = 0; i < t->children.size(); i++)
        {
            deleteChildren(t->children[i]);
        }

        int sceneObjIndex = getSceneObjectIndex(t->sceneObject);
        Odysseus::SceneObject* obj = _objectsInScene[sceneObjIndex];
        _objectsInScene.erase(_objectsInScene.begin() + sceneObjIndex);
        delete obj;
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
        this->updateEditorScene();

        // TODO: Find a way to stop the scene when paused -> probably by using SceneManager time & SceneState or by creating some other funcitons
        if (this->isRuntimeScene && this->status == SceneState::RUNNING)
            this->updateRuntimeScene();
    }

}