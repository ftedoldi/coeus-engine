#ifndef __CAMERAMOVEMENT_H__
#define __CAMERAMOVEMENT_H__

#include <Window.hpp>

#include <Camera.hpp>
#include <Behaviour.hpp>
#include <Vector3.hpp>
#include <Math.hpp>
#include <Scalar.hpp>

class CameraMovement : public System::Behaviour { 
    public:
        Odysseus::Camera* camera;

        Athena::Scalar xMousePos;
        Athena::Scalar yMousePos;

        Athena::Scalar movementSpeed;
        Athena::Scalar mouseSensitivity;

        Athena::Scalar yaw;
        Athena::Scalar pitch;

        CameraMovement();

        virtual void start();
        virtual void update();

        virtual void setOrderOfExecution(const short& newOrderOfExecution);

        virtual short getUniqueID();
        virtual std::string toString();

        virtual ~CameraMovement();
};

#endif // __CAMERAMOVEMENT_H__