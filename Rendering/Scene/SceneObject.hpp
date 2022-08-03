#ifndef __SCENEOBJECT_H__
#define __SCENEOBJECT_H__

#include <Dockspace.hpp>

#include <UUID.hpp>

#include "Component.hpp"
#include "Container.hpp"
#include <Transform.hpp>
#include <SceneGraph.hpp>

#include <vector>
#include <string>
#include <iostream>
#include <ctime>

namespace System {
    class Component;
    class Dockspace;
    class UUID;
}

namespace Odysseus
{
    class Container;
    class SceneGraph;

    class SceneObject {
        friend class SceneGraph;
        friend class System::Dockspace;

        private:
            bool _active;
            bool _static;
            Container* _container;

        public:
            System::UUID ID;

            Transform* transform;

            std::string tag;
            std::string layer;

            const bool& isActive;
            const bool& isStatic;

            SceneObject();
            SceneObject(const std::string& name);
            SceneObject(const Transform& transform);
            SceneObject(const Transform& transform, const std::string& name);
            SceneObject(const SceneObject& sceneObject);

            static SceneObject* FindSceneObjectWithName(const std::string& name);
            static SceneObject* FindSceneObjectWitTag(const std::string& tag);

            template<class T> T* findSceneObjectWithComponent() {
                for (int i = 0; i < SceneGraph::objectsInScene.size(); i++)
                    if (SceneGraph::objectsInScene[i]->getComponent<T>() != nullptr)
                        return SceneGraph::objectsInScene[i];

                return nullptr;
            }

            template<class T> std::vector<T*> findSceneObjectsWithComponent() {
                std::vector<T*> objects;

                for (int i = 0; i < SceneGraph::objectsInScene.size(); i++)
                    if (SceneGraph::objectsInScene[i]->getComponent<T>() != nullptr)
                        objects.push_back(SceneGraph::objectsInScene[i]);

                return objects;
            }

            template<class T> T* getComponent() {
                T* instance = new T();

                for (int i = 0; i < _container->_components.size(); i++)
                    if (_container->_components[i]->toString() == instance->toString())
                        return dynamic_cast<T*>(_container->_components[i]);

                delete instance;

                return nullptr;
            }

            template<class T> std::vector<T*> getComponents() {
                T* instance = new T();
                std::vector<T*> comps;

                for (int i = 0; i < _container->_components.size(); i++)
                    if (_container->_components[i]->toString() == instance->toString())
                        comps.push_back(dynamic_cast<T*>(_container->_components[i]));

                delete instance;

                return comps;
            }

            template<class T> T* addCopyOfExistingComponent(T* component) {
                if (component == nullptr) {
                    std::cerr << "WARNING:: In SceneObject " << this->transform->name 
                        << " -> in method addCopyOfExistingComponent you passed a null pointer to a component!" << std::endl;
                    return nullptr;
                }

                component->sceneObject = this;
                component->transform = this->transform;

                _container->_components.push_back(component);

                return dynamic_cast<T*>(_container->_components[_container->_components.size() - 1]);
            }

            template<class T> T* addComponent() {
                T* instance = new T();

                instance->sceneObject = this;
                instance->transform = this->transform;

                _container->_components.push_back(instance);

                return dynamic_cast<T*>(_container->_components[_container->_components.size() - 1]);
            }
            
            template<class T> bool removeComponent() {
                T* instance = new T();

                for (int i = 0; i < _container->components.size(); i++)
                    if (_container->components[i]->toString() == instance->toString()) {
                        _container->_components.erase(_container->_components.begin() + i);
                        return true;
                    }

                delete instance;

                return false;
            }

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