#include "../SceneObject.hpp"

namespace Odysseus {

    SceneObject::SceneObject() : isActive(_active), isStatic(_static)
    {
        this->transform = new Transform();
        this->container = new Container(*this, *this->transform);

        this->ID = SceneGraph::objectsInScene.size() + 1;

        SceneGraph::objectsInScene.push_back(this);
    }

    SceneObject::SceneObject(const std::string& name) : isActive(_active), isStatic(_static)
    {
        this->transform = new Transform();
        this->container = new Container(*this, *this->transform);

        this->ID = SceneGraph::objectsInScene.size() + 1;

        SceneGraph::objectsInScene.push_back(this);
    }

    SceneObject::SceneObject(const Transform& transform) : isActive(_active), isStatic(_static)
    {
        this->transform = new Transform(transform);
        this->container = new Container(*this, *this->transform);

        this->ID = SceneGraph::objectsInScene.size() + 1;

        SceneGraph::objectsInScene.push_back(this);
    }

    SceneObject::SceneObject(const Transform& transform, const std::string& name) : isActive(_active), isStatic(_static)
    {
        this->transform = new Transform(transform);
        this->container = new Container(*this, *this->transform);

        this->ID = SceneGraph::objectsInScene.size() + 1;

        SceneGraph::objectsInScene.push_back(this);
    }

    SceneObject::SceneObject(const SceneObject& sceneObject) : isActive(_active), isStatic(_static)
    {
        this->transform = new Transform(*sceneObject.transform);
        this->container = new Container(*this, *this->transform);

        this->ID = SceneGraph::objectsInScene.size() + 1;

        SceneGraph::objectsInScene.push_back(this);
    }

    bool SceneObject::operator == (const SceneObject& object) const
    {
        return object.container == this->container;
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
        delete container;
        delete transform;
    }

}