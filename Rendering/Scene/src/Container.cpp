#include <Container.hpp>

namespace Odysseus {

    Container::Container(SceneObject& owner, Transform& ownerTransform) : components(_components)
    {

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
        for (int i = this->_components.size() - 1; i >= 0; i--)
        {
            delete this->_components[i];
        }
    }

}