#include "../SceneObject.hpp"

namespace Odysseus {

    SceneObject::SceneObject() : isActive(_active), isStatic(_static)
    {
        this->transform = new Transform();
        this->_container = new Container(*this, *this->transform);

        this->ID = static_cast<short>(std::time(0));

        SceneGraph::objectsInScene.push_back(this);
    }

    SceneObject::SceneObject(const std::string& name) : isActive(_active), isStatic(_static)
    {
        this->transform = new Transform();
        this->_container = new Container(*this, *this->transform);

        this->ID = static_cast<short>(std::time(0));

        SceneGraph::objectsInScene.push_back(this);
    }

    SceneObject::SceneObject(const Transform& transform) : isActive(_active), isStatic(_static)
    {
        this->transform = new Transform(transform);
        this->_container = new Container(*this, *this->transform);

        this->ID = static_cast<short>(std::time(0));

        SceneGraph::objectsInScene.push_back(this);
    }

    SceneObject::SceneObject(const Transform& transform, const std::string& name) : isActive(_active), isStatic(_static)
    {
        this->transform = new Transform(transform);
        this->_container = new Container(*this, *this->transform);

        this->ID = static_cast<short>(std::time(0));

        SceneGraph::objectsInScene.push_back(this);
    }

    SceneObject::SceneObject(const SceneObject& sceneObject) : isActive(_active), isStatic(_static)
    {
        this->transform = new Transform(*sceneObject.transform);
        this->_container = new Container(*this, *this->transform);

        this->ID = static_cast<short>(std::time(0));

        SceneGraph::objectsInScene.push_back(this);
    }

    SceneObject* SceneObject::FindSceneObjectWithName(const std::string& name) {
        for (int i = 0; i < SceneGraph::objectsInScene.size(); i++) {
            if (SceneGraph::objectsInScene[i]->transform->name == name)
                return SceneGraph::objectsInScene[i];   
        }

        return nullptr;
    }

    SceneObject* SceneObject::FindSceneObjectWitTag(const std::string& tag) {
        for (int i = 0; i < SceneGraph::objectsInScene.size(); i++) {
            if (SceneGraph::objectsInScene[i]->tag == tag)
                return SceneGraph::objectsInScene[i];   
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

    void SceneObject::destroy()
    {
        SceneGraph::objectsInScene.erase(SceneGraph::objectsInScene.begin() + this->ID);
        delete this;
    }

    SceneObject::~SceneObject()
    {
        delete _container;
        delete transform;
    }

}