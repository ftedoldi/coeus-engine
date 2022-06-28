#include <SceneObject.hpp>

namespace Odysseus {

    std::vector<SceneObject*> SceneObject::objectsInScene;

    SceneObject::SceneObject() : isActive(_active), isStatic(_static), components(_components)
    {

    }

    SceneObject::SceneObject(const std::string& name) : isActive(_active), isStatic(_static), components(_components)
    {

    }

    SceneObject::SceneObject(const Transform& transform) : isActive(_active), isStatic(_static), components(_components)
    {

    }

    SceneObject::SceneObject(const Transform& transform, const std::string& name) : isActive(_active), isStatic(_static), components(_components)
    {

    }

    SceneObject::SceneObject(const SceneObject& sceneObject) : isActive(_active), isStatic(_static), components(_components)
    {

    }

}