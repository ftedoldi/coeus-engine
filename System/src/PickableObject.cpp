#include "../PickableObject.hpp"

#include <SceneObject.hpp>

namespace System::Picking 
{
    std::unordered_map<float, Odysseus::SceneObject*> PickableObject::pickableObjects;

    void PickableObject::insertPickableObject(const float& id, Odysseus::SceneObject* pickableMesh)
    {
        pickableObjects[id] = pickableMesh;
    }

    bool PickableObject::getPickableObject(const float& id, Odysseus::SceneObject** outSceneObject)
    {
        if (pickableObjects.find(id) == pickableObjects.end())
            return false;

        *outSceneObject = pickableObjects[id];
        return true;
    }

    bool PickableObject::removePickableObject(const float& id)
    {
        if (pickableObjects.find(id) == pickableObjects.end())
            return false;

        pickableObjects.erase(id);
        return true;
    }

}