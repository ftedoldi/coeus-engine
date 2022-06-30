#include <Container.hpp>

namespace Odysseus {

    Container::Container(SceneObject& owner, Transform& ownerTransform) : components(_components), _owner(owner), _ownerTransform(ownerTransform)
    {

    }

    void Container::addComponent(Component& component)
    {
        component.sceneObject = &_owner;
        component.transform = &_ownerTransform;
        _components.push_back(&component);
    }

    Component* Container::getComponent(Component& component)
    {
        for (int i = 0; i < components.size(); i++)
            if (components[i]->getUniqueID() == component.getUniqueID())
                return components[i];
        
        return nullptr;
    }

    void Container::removeComponent(Component& component)
    {
        for (int i = 0; i < _components.size(); i++)
            if (_components[i]->getUniqueID() == component.getUniqueID()) {
                _components.erase(components.begin() + i);
                return;
            }
    }

    bool Container::operator == (Container& b)
    {
        if (b.components.size() != this->components.size())
            return false;

        for (int i = 0; i < components.size(); i++)
            if (this->components[i]->getUniqueID() != b.components[i]->getUniqueID())
                return false;

        return true;
    }

    Container::~Container()
    {

    }

}