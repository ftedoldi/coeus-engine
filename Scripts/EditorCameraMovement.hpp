#ifndef __EDITORCAMERAMOVEMENT_H__
#define __EDITORCAMERAMOVEMENT_H__

#include <Behaviour.hpp>
#include <Vector3.hpp>
#include <Math.hpp>
#include <Scalar.hpp>
#include <Shader.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace Odysseus
{
    class EditorCamera;
}

class EditorCameraMovement : public System::Behaviour { 
    public:
        Odysseus::EditorCamera* editorCamera;
        Odysseus::Shader* shader;

        Athena::Scalar xMousePos;
        Athena::Scalar yMousePos;

        Athena::Scalar movementSpeed;
        Athena::Scalar mouseSensitivity;

        Athena::Scalar yaw;
        Athena::Scalar pitch;

        Athena::Scalar yValue;

        EditorCameraMovement();

        virtual void start();
        virtual void update();

        virtual void setOrderOfExecution(const short& newOrderOfExecution);

        virtual int getUniqueID();
        virtual std::string toString();

        void setShader(Odysseus::Shader* shader);
        virtual ~EditorCameraMovement();
};

#endif // __EDITORCAMERAMOVEMENT_H__