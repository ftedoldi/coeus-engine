#ifndef __PICKABLEOBJECT_H__
#define __PICKABLEOBJECT_H__

#include <SceneObject.hpp>

#include <unordered_map>

namespace Odysseus 
{
    class SceneObject;
    class Mesh;
}

namespace System
{
    namespace Picking
    {
        class Odysseus::SceneObject;
        class Odysseus::Mesh;

        class PickableObject 
        {
            private:
                static std::unordered_map<float, Odysseus::SceneObject*> pickableObjects;
            
            public:
                static void insertPickableObject(const float& id, Odysseus::SceneObject* pickableMesh);
                static bool getPickableObject(const float& id, Odysseus::SceneObject** outSceneObject);
                static bool removePickableObject(const float& id);
        };
    } // namespace Picking
} // namespace System

#endif // __PICKABLEOBJECT_H__