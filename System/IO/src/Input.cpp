#include <IO/Input.hpp>

namespace System {
    Mouse Input::mouse;
    Keyboard* Input::keyboard;

    Keyboard::Keyboard()
    {
        _pressedKey = GLFW_KEY_UNKNOWN;

        auto keyboardCallback = []( GLFWwindow* window, int key, int scancode, int action, int mods ) {
            if (action == GLFW_PRESS)
                Input::keyboard->_pressedKey = key;

            if (action == GLFW_RELEASE)
                Input::keyboard->_pressedKey = GLFW_KEY_UNKNOWN;
        };

        glfwSetKeyCallback(Window::window, keyboardCallback);
    }

    int Keyboard::getPressedKey()
    {
        return _pressedKey;
    }

}