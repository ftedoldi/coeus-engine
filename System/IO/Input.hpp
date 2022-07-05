#ifndef __INPUT_H__
#define __INPUT_H__

#include <Scalar.hpp>

namespace System
{
    struct Mouse
    {
        bool isFirstMovement;

        Athena::Scalar xPosition;
        Athena::Scalar yPosition;

        Athena::Scalar xOffsetFromLastPosition;
        Athena::Scalar yOffsetFromLastPosition;
    };
    
    // TODO: Implement method in order to get a key pressed event
    class Input {
        public:
            static Mouse mouse;
    };
}

#endif // __INPUT_H__