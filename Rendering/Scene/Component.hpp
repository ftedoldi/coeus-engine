#ifndef __COMPONENT_H__
#define __COMPONENT_H__

#include <vector>

namespace Odysseus {
    class Component {
        public:
            Component();

            template<class T> T* getComponentType();
    };
}

#endif // __COMPONENT_H__