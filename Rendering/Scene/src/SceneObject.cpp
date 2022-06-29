#include "../SceneObject.hpp"

namespace Odysseus {

    SceneObject::SceneObject() : isActive(_active), isStatic(_static)
    {

    }

    SceneObject::SceneObject(const std::string& name) : isActive(_active), isStatic(_static)
    {

    }

    SceneObject::SceneObject(const Transform& transform) : isActive(_active), isStatic(_static)
    {

    }

    SceneObject::SceneObject(const Transform& transform, const std::string& name) : isActive(_active), isStatic(_static)
    {

    }

    SceneObject::SceneObject(const SceneObject& sceneObject) : isActive(_active), isStatic(_static)
    {

    }

    bool SceneObject::operator == (const SceneObject& object) const
    {
        return false;
    }

    bool SceneObject::operator != (const SceneObject& object) const
    {
        return false;
    }

    SceneObject SceneObject::clone() const
    {
        return SceneObject();
    }

    void SceneObject::setActive(const bool& newState)
    {

    }

    void SceneObject::setStatic(const bool& newState)
    {

    }

    void SceneObject::destroy()
    {

    }

    SceneObject::~SceneObject()
    {

    }

}