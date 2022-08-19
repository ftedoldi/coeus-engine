#include "../SceneObject.hpp"

namespace Odysseus {

    SceneObject::SceneObject() : isActive(_active), isStatic(_static)
    {
        this->transform = new Transform();
        this->_container = new Container(*this, *this->transform);

        this->transform->sceneObject = this;
 
        this->ID = System::UUID();

        this->showInEditor = true;
 
        SceneManager::activeScene->addSceneObject(this);
    }

    SceneObject::SceneObject(const std::string& name) : isActive(_active), isStatic(_static)
    {
        this->transform = new Transform();
        this->_container = new Container(*this, *this->transform);

        this->transform->sceneObject = this;
 
        this->ID = System::UUID();
 
        this->showInEditor = true;
 
        SceneManager::activeScene->addSceneObject(this);
    }

    SceneObject::SceneObject(const Transform& transform) : isActive(_active), isStatic(_static)
    {
        this->transform = new Transform(transform);
        this->_container = new Container(*this, *this->transform);

        this->transform->sceneObject = this;
 
        this->ID = System::UUID();
 
        this->showInEditor = true;
 
        SceneManager::activeScene->addSceneObject(this);
    }

    SceneObject::SceneObject(const Transform& transform, const std::string& name) : isActive(_active), isStatic(_static)
    {
        this->transform = new Transform(transform);
        this->_container = new Container(*this, *this->transform);

        this->transform->name = name;

        this->transform->sceneObject = this;
 
        this->ID = System::UUID();
 
        this->showInEditor = true;
 
        SceneManager::activeScene->addSceneObject(this);
    }

    SceneObject::SceneObject(const Transform& transform, const std::uint64_t& uuid) : isActive(_active), isStatic(_static)
    {
        this->transform = new Transform(transform);
        this->_container = new Container(*this, *this->transform);

        this->transform->sceneObject = this;
 
        this->ID = uuid;
 
        this->showInEditor = true;
 
        SceneManager::activeScene->addSceneObject(this);
    }

    SceneObject::SceneObject(const SceneObject& sceneObject) : isActive(_active), isStatic(_static)
    {
        this->transform = new Transform(*sceneObject.transform);
        this->_container = new Container(*this, *this->transform);

        this->transform->sceneObject = this;
 
        this->ID = System::UUID();
 
        this->showInEditor = true;
 
        SceneManager::activeScene->addSceneObject(this);
    }

    SceneObject* SceneObject::FindSceneObjectWithName(const std::string& name) {
        for (int i = 0; i < SceneManager::activeScene->objectsInScene.size(); i++) {
            if (SceneManager::activeScene->objectsInScene[i]->transform->name == name)
                return SceneManager::activeScene->objectsInScene[i];   
        }

        return nullptr;
    }

    SceneObject* SceneObject::FindSceneObjectWitTag(const std::string& tag) {
        for (int i = 0; i < SceneManager::activeScene->objectsInScene.size(); i++) {
            if (SceneManager::activeScene->objectsInScene[i]->tag == tag)
                return SceneManager::activeScene->objectsInScene[i];   
        }

        return nullptr;
    }

    bool SceneObject::operator == (const SceneObject& object) const
    {
        return object._container == this->_container;
    }

    bool SceneObject::operator != (const SceneObject& object) const
    {
        return !(*this == object);
    }

    SceneObject SceneObject::clone() const
    {
        return SceneObject(*this);
    }

    void SceneObject::setActive(const bool& newState)
    {
        _active = newState;
    }

    void SceneObject::setStatic(const bool& newState)
    {
        _static = newState;
    }

    // FIXME: Erease correctly form game scene, right now we erease the first element, we want to erase the i element
    void SceneObject::destroy()
    {
        SceneManager::activeScene->deleteSceneObject(this);
        delete this;
    }

    SceneObject::~SceneObject()
    {
        delete _container;
        delete transform;
    }

}