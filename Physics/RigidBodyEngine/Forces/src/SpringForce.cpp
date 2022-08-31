#include "../SpringForce.hpp"

namespace Khronos
{
    SpringForce::SpringForce(const Athena::Vector3& localConnectionPoint,
                             RigidBody* other,
                             const Athena::Vector3& otherConnectionPoint,
                             Athena::Scalar springConstant,
                             Athena::Scalar restLength)
    : connectionPoint(localConnectionPoint), other(other), otherConnectionPoint(otherConnectionPoint),
      springConstant(springConstant), restLength(restLength)
    {
        
    }

    void SpringForce::updateForce(RigidBody* body, Athena::Scalar dt)
    {
        //firstly calculate the two ends of the spring in world space
        Athena::Vector3 bodyPointWS = body->getPointInWorldSpace(this->connectionPoint);
        Athena::Vector3 otherPointWS = this->other->getPointInWorldSpace(this->otherConnectionPoint);

        //calculate the vector of the spring
        Athena::Vector3 force = bodyPointWS - otherPointWS;

        //calculate the magnitute of the force
        Athena::Scalar magnitude = force.magnitude();
        magnitude = Athena::Math::scalarAbs(magnitude - this->restLength);
        magnitude *= this->springConstant;

        //calculate the final force and apply it
        force.normalize();
        force *= -magnitude;
        body->addForceAtPoint(force, bodyPointWS);
    }
}