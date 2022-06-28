#ifndef __SCENEOBJECT_H__
#define __SCENEOBJECT_H__

#include "Component.hpp"
#include <Transform.hpp>

#include <string>

namespace Odysseus
{
    class SceneObject {
        private:
            bool _active;
            bool _static;
            std::vector<Component> _components;

            static std::vector<SceneObject*> objectsInScene;

        public:
            Transform* transform;
            const std::vector<Component>& components;

            std::string tag;
            std::string layer;

            const bool& isActive;
            const bool& isStatic;

            SceneObject();
            SceneObject(const std::string& name);
            SceneObject(const Transform& transform);
            SceneObject(const Transform& transform, const std::string& name);
            SceneObject(const SceneObject& sceneObject);

            // static SceneObject* FindSceneObjectWithName(const std::string& name);
            // template<class T> static SceneObject* FindSceneObjectOfType(T type);
            // template<class T> static std::vector<SceneObject*> FindSceneObjectsOfType(T type);

            // template<class T> T* getComponentObject();
            // template<class T> T* addComponentObject(const T& component);
            // template<class T> T* addComponentObject();
            // template<class T> T* removeComponentObject();

            bool operator == (const SceneObject& object) const;
            bool operator != (const SceneObject& object) const;

            SceneObject clone() const;

            void setActive(const bool& newState);
            void setStatic(const bool& newState);

            void destroy();

            ~SceneObject();
    };
} // namespace Odysseus

#endif // __SCENEOBJECT_H__