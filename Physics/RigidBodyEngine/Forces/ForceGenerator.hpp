#ifndef FORCEGENERATOR_HPP
#define FORCEGENERATOR_HPP
#include <RigidBody.hpp>
#include <Scalar.hpp>

namespace Khronos
{
    class ForceGenerator
    {
    public:
        virtual void updateForce(RigidBody* body, Athena::Scalar dt) = 0;
    };
}

#endif