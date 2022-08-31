#ifndef RIGIDBODY_HPP
#define RIGIDBODY_HPP
#define SCALAR_MAX FLT_MAX
#include <Scalar.hpp>
#include <Vector3.hpp>
#include <Matrix3.hpp>
#include <Matrix4.hpp>
#include <Quaternion.hpp>

namespace Khronos
{
    class RigidBody
    {
    public:

        //holds the inverse mass of the rigid body
        Athena::Scalar inverseMass;

        //holds the position of the rigid body
        Athena::Vector3 position;

        //holds the angular orientation of the rigid body in world space
        Athena::Quaternion orientation;

        //holds the linear velocity of the rigid body
        Athena::Vector3 velocity;

        //holds the angular velocity (rotation) of the rigid body
        Athena::Vector3 rotation;
        
        //holds a transform matrix for converting to object space to world space
        Athena::Matrix4 transformMatrix;

        //holds the inverse of the rigid body inertia tensor in local space
        Athena::Matrix3 inverseInertiaTensor;

        //holds the inverse of the inertia tensor in world space
        Athena::Matrix3 inverseInertiaTensorWorld;

        //holds the sum of all forces (since from D'Alembert principle the effect
        //of one accumulated force is identical to the effect of all its component forces)
        Athena::Vector3 forceAccum;

        //holds the sum of all torque forces
        //if a force always apply on the center of mass we dont work out its torque component
        //since it will never induce rotation.
        Athena::Vector3 torqueAccum;

        //holds the amount of dampung applied to the linear motion
        Athena::Scalar linearDamping;

        //just like the linear damping, the angular damping holds the amount of damping
        //applied to the angular motion
        Athena::Scalar angularDamping;

        //Holds the acceleration of the rigid body
        Athena::Vector3 acceleration;

        //Holds the acceleration at last frame of the rigid body
        Athena::Vector3 lastFrameAcceleration;

        bool hasFiniteMass() const;
        Athena::Scalar getMass() const;
        Athena::Scalar getInverseMass() const;
        void setMass(const Athena::Scalar mass);
        void setInverseMass(const Athena::Scalar inverseMass);

        void setInertiaTensor(const Athena::Matrix3& inertiaTensor);
        Athena::Matrix3 getInertiaTensor() const;
        Athena::Matrix3 getInertiaTensorWorld() const;

        void setInverseInertiaTensor(const Athena::Matrix3& inverseInertiaTensor);
        Athena::Matrix3 getInverseInertiaTensor() const;
        Athena::Matrix3 getInverseInertiaTensorWorld() const;

        void setDamping(const Athena::Scalar linearDamping, const Athena::Scalar angularDamping);

        void setLinearDamping(const Athena::Scalar linearDamping);
        Athena::Scalar getLinearDamping() const;

        void setAngularDamping(const Athena::Scalar angularDamping);
        Athena::Scalar getAngularDamping() const;

        void setPosition(const Athena::Vector3& pos);
        void setPosition(const Athena::Scalar x, const Athena::Scalar y, const Athena::Scalar z);
        Athena::Vector3 getPosition() const;

        void setOrientation(const Athena::Quaternion& quat);
        void setOrientation(const Athena::Scalar real, const Athena::Scalar x, const Athena::Scalar y, const Athena::Scalar z);
        void setOrientation(const Athena::Scalar real, const Athena::Vector3& imm);
        Athena::Quaternion getOrientation() const;

        void setVelocity(const Athena::Vector3& velocity);
        void setVelocity(const Athena::Scalar x, const Athena::Scalar y, const Athena::Scalar z);
        Athena::Vector3 getVelocity() const;

        void calculateDerivedData();

        void addForce(const Athena::Vector3& force);
        void addVelocity(const Athena::Vector3& velocity);
        void addTorque(const Athena::Vector3& torque);
        void addForceAtBodyPoint(const Athena::Vector3& force, const Athena::Vector3& point);
        void addForceAtPoint(const Athena::Vector3& force, const Athena::Vector3& point);

        Athena::Vector3 getPointInWorldSpace(const Athena::Vector3& point);

        //every frame we clear the torque and force accumulators
        void clearAccumulators();

        void integrate(Athena::Scalar dt);

    };
}

#endif