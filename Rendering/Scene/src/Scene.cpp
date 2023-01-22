#include "../Scene.hpp"

#include <Window.hpp>

#include <Component.hpp>
#include <Mesh.hpp>
#include <RigidBox.hpp>
#include <RigidPlane.hpp>
#include <RigidSphere.hpp>

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

    Scene::~Scene()
    {
        delete this->physicSimulation;
        this->physicSimulation = nullptr;
    }

    void Scene::initialiseEditorScene()
    {
        for (int i = 0; i < _objectsInScene.size(); i++)
        {
            for (int k = 0; k < _objectsInScene[i]->_container->components.size(); k++)
            {
                _objectsInScene[i]->_container->components[k]->start();
                glUseProgram(0);
            }
        }
    }

    void Scene::initialiseRuntimeScene()
    {
        this->physicSimulation = new Khronos::RigidPhysicsEngine();
        for (int i = 0; i < _objectsInScene.size(); i++)
        {
            for (int k = 0; k < _objectsInScene[i]->_container->components.size(); k++)
            {
                if(_objectsInScene[i]->_container->components[k]->toString() == "RigidBox")
                   {
                        auto rigidBoxComponent = dynamic_cast<Khronos::RigidBox*>(_objectsInScene[i]->_container->components[k]);
                        rigidBoxComponent->setPhysicsSimulation(this->physicSimulation);
                   }
                if(_objectsInScene[i]->_container->components[k]->toString() == "RigidPlane")
                    {
                        auto rigidPlaneComponent = dynamic_cast<Khronos::RigidPlane*>(_objectsInScene[i]->_container->components[k]);
                        rigidPlaneComponent->setPhysicsSimulation(this->physicSimulation);
                    }
                if(_objectsInScene[i]->_container->components[k]->toString() == "RigidSphere")
                    {
                        auto rigidSphereComponent = dynamic_cast<Khronos::RigidSphere*>(_objectsInScene[i]->_container->components[k]);
                        rigidSphereComponent->setPhysicsSimulation(this->physicSimulation);
                    }
                _objectsInScene[i]->_container->components[k]->startRuntime();

                glUseProgram(0);
            }
        }
    }

    void Scene::updateEditorScene()
    {
        for (int i = 0; i < _objectsInScene.size(); i++)
        {
            for (int j = 0; j < _objectsInScene[i]->_container->components.size(); j++)
            {
                _objectsInScene[i]->_container->components[j]->update();
                glUseProgram(0);
            }
        }
    }

    void Scene::updateRuntimeScene()
    {
        for (int i = 0; i < _objectsInScene.size(); i++)
        {
            for (int j = 0; j < _objectsInScene[i]->_container->components.size(); j++)
            {
                _objectsInScene[i]->_container->components[j]->updateRuntime();
                glUseProgram(0);
            }
        }
        physicSimulation->update();
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

        if (this->isRuntimeScene && this->status == SceneState::RUNNING)
            this->updateRuntimeScene();
    }

}