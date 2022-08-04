#ifndef __INPUT_H__
#define __INPUT_H__

#include <Window.hpp>

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

    class Keyboard
    {
        friend class Input;

        private:
            int _pressedKey;

        public:
            Keyboard();

            int getPressedKey();
    };
    
    // TODO: Implement method in order to get a key pressed event
    class Input {
        public:
            static Mouse mouse;
            static Keyboard* keyboard;
    };
}

#endif // __INPUT_H__