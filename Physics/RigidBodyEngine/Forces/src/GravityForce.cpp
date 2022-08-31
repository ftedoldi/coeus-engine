#include "../GravityForce.hpp"

namespace Khronos
{
    GravityForce::GravityForce(const Athena::Vector3& gravity) : gravity(gravity)
    {

    }

    void GravityForce::updateForce(RigidBody* body, Athena::Scalar dt)
    {
        if(!body->hasFiniteMass())
            return;
        
        body->addForce(this->gravity * body->getMass());
    }
}