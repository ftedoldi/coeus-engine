#ifndef GRAVITYFORCE_HPP
#define GRAVITYFORCE_HPP
#include <ForceGenerator.hpp>

namespace Khronos
{
    class GravityForce : public ForceGenerator
    {
        //holds the acceleration due to gravity
        Athena::Vector3 gravity;

    public:
        GravityForce(const Athena::Vector3& gravity);

        virtual void updateForce(RigidBody* body, Athena::Scalar dt);

    };
}

#endif