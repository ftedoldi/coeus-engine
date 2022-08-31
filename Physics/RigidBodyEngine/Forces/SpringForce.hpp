#ifndef SPRINGFORCE_HPP
#define SPRINGFORCE_HPP
#include <ForceGenerator.hpp>
#include <Vector3.hpp>
#include <Math.hpp>

namespace Khronos
{
    class SpringForce: public ForceGenerator
    {
        //holds the position of the point of connection of the spring (local coordinates)
        Athena::Vector3 connectionPoint;

        //holds the position of the point of connection of the spring to the other object
        //(local coordinates)
        Athena::Vector3 otherConnectionPoint;

        //the rigid body at the other end of the spring
        RigidBody* other;

        //holds the spring constant
        Athena::Scalar springConstant;

        //hold the length of the spring when rested
        Athena::Scalar restLength;

    public:

        //Create a new spring with the given parameters
        SpringForce(const Athena::Vector3& localConnectionPoint,
                    RigidBody* other,
                    const Athena::Vector3& otherConnectionPoint,
                    Athena::Scalar springConstant,
                    Athena::Scalar restLength);
        
        //applies the spring force to a given rigid body
        virtual void updateForce(RigidBody* body, Athena::Scalar dt);
    };
}

#endif