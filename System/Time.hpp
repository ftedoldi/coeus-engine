#ifndef __ENGINE_TIME_H__
#define __ENGINE_TIME_H__

#include <Scalar.hpp>
#include <SceneGraph.hpp>

namespace System {
    class Time {
        public:
            static Athena::Scalar deltaTime;
            static Athena::Scalar timeAtLastFrame;
            static Athena::Scalar time;
    };
}

#endif // __ENGINE_TIME_H__