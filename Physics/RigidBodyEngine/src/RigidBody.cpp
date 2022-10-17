#include "../RigidBody.hpp"

namespace Khronos
{

    //creates a transform matrix from a position and orientation
    //firstly we convert a quaternion to a rotation matrix
    //then, the last column contains the translation (position) vector
    static inline void calculateTransformMatrix(Athena::Matrix4& transformMatrix, const Athena::Vector3& position, const Athena::Quaternion &orientation)
    {
        transformMatrix.data[0] = 1.0f - 2.0f * orientation.immaginary.coordinates.y * orientation.immaginary.coordinates.y - 
                                    2.0f * orientation.immaginary.coordinates.z * orientation.immaginary.coordinates.z;

        transformMatrix.data[1] = 2.0f * orientation.immaginary.coordinates.x * orientation.immaginary.coordinates.y -
                                    2.0f * orientation.real * orientation.immaginary.coordinates.z;

        transformMatrix.data[2] = 2.0f * orientation.immaginary.coordinates.x * orientation.immaginary.coordinates.z +
                                    2.0f * orientation.real * orientation.immaginary.coordinates.y;

        transformMatrix.data[3] = position.coordinates.x;

        transformMatrix.data[4] = 2.0f * orientation.immaginary.coordinates.x * orientation.immaginary.coordinates.y +
                                    2.0f * orientation.real * orientation.immaginary.coordinates.z;

        transformMatrix.data[5] = 1.0f - 2.0f * orientation.immaginary.coordinates.x * orientation.immaginary.coordinates.x - 
                                    2.0f * orientation.immaginary.coordinates.z * orientation.immaginary.coordinates.z;

        transformMatrix.data[6] = 2.0f * orientation.immaginary.coordinates.y * orientation.immaginary.coordinates.z -
                                    2.0f * orientation.real * orientation.immaginary.coordinates.x;

        transformMatrix.data[7] = position.coordinates.y;

        transformMatrix.data[8] = 2.0f * orientation.immaginary.coordinates.x * orientation.immaginary.coordinates.z -
                                    2.0f * orientation.real * orientation.immaginary.coordinates.y;

        transformMatrix.data[9] = 2.0f * orientation.immaginary.coordinates.y * orientation.immaginary.coordinates.z +
                                    2.0f * orientation.real * orientation.immaginary.coordinates.x;

        transformMatrix.data[10] = 1.0f - 2.0f * orientation.immaginary.coordinates.x * orientation.immaginary.coordinates.x -
                                    2.0f * orientation.immaginary.coordinates.y * orientation.immaginary.coordinates.y;

        transformMatrix.data[11] = position.coordinates.z;
    }

    /**
     * Since we need to have the inertia tensor matrix in world space
     * we need at each frame to convert the constant inertia tensor
     * in object coordinates into the corresponding matrix in world
     * coordinates
    */
    static inline void transformInertiaTensor(Athena::Matrix3& iitWorld,
                                                const Athena::Quaternion& q,
                                                const Athena::Matrix3& iitBody,
                                                const Athena::Matrix4& rotMatrix)
    {
        Athena::Scalar t4 = rotMatrix.data[0] * iitBody.data[0] +
            rotMatrix.data[1] * iitBody.data[3] +
            rotMatrix.data[2] * iitBody.data[6];

        Athena::Scalar t9 = rotMatrix.data[0] * iitBody.data[1] +
            rotMatrix.data[1] * iitBody.data[4] +
            rotMatrix.data[2] * iitBody.data[7];

        Athena::Scalar t14 = rotMatrix.data[0] * iitBody.data[2] +
            rotMatrix.data[1] * iitBody.data[5] +
            rotMatrix.data[2] * iitBody.data[8];

        Athena::Scalar t28 = rotMatrix.data[4] * iitBody.data[0] +
            rotMatrix.data[5] * iitBody.data[3] +
            rotMatrix.data[6] * iitBody.data[6];

        Athena::Scalar t33 = rotMatrix.data[4] * iitBody.data[1] +
            rotMatrix.data[5] * iitBody.data[4] +
            rotMatrix.data[6] * iitBody.data[7];

        Athena::Scalar t38 = rotMatrix.data[4] * iitBody.data[2] +
            rotMatrix.data[5] * iitBody.data[5] +
            rotMatrix.data[6] * iitBody.data[8];

        Athena::Scalar t52 = rotMatrix.data[8] * iitBody.data[0] +
            rotMatrix.data[9] * iitBody.data[3] +
            rotMatrix.data[10] * iitBody.data[6];

        Athena::Scalar t57 = rotMatrix.data[8] * iitBody.data[1] +
            rotMatrix.data[9] * iitBody.data[4] +
            rotMatrix.data[10] * iitBody.data[7];

        Athena::Scalar t62 = rotMatrix.data[8] * iitBody.data[2] +
            rotMatrix.data[9] * iitBody.data[5] +
            rotMatrix.data[10] * iitBody.data[8];


        iitWorld.data[0] = t4 * rotMatrix.data[0] +
            t9 * rotMatrix.data[1] +
            t14 * rotMatrix.data[2];

        iitWorld.data[1] = t4 * rotMatrix.data[4] +
            t9 * rotMatrix.data[5] +
            t14 * rotMatrix.data[6];

        iitWorld.data[2] = t4 * rotMatrix.data[8] +
            t9 * rotMatrix.data[9] +
            t14 * rotMatrix.data[10];

        iitWorld.data[3] = t28 * rotMatrix.data[0] +
            t33 * rotMatrix.data[1] +
            t38 * rotMatrix.data[2];

        iitWorld.data[4] = t28 * rotMatrix.data[4] +
            t33 * rotMatrix.data[5] +
            t38 * rotMatrix.data[6];

        iitWorld.data[5] = t28 * rotMatrix.data[8] +
            t33 * rotMatrix.data[9] +
            t38 * rotMatrix.data[10];

        iitWorld.data[6] = t52 * rotMatrix.data[0] +
            t57 * rotMatrix.data[1] +
            t62 * rotMatrix.data[2];

        iitWorld.data[7] = t52 * rotMatrix.data[4] +
            t57 * rotMatrix.data[5] +
            t62 * rotMatrix.data[6];

        iitWorld.data[8] = t52 * rotMatrix.data[8] +
            t57 * rotMatrix.data[9] +
            t62 * rotMatrix.data[10];
    }

    bool RigidBody::hasFiniteMass() const
    {
        return this->inverseMass >= 0.0f;
    }

    Athena::Scalar RigidBody::getMass() const
    {
        if(this->inverseMass == 0)
            return SCALAR_MAX;
        else
            return ((Athena::Scalar)1.0f) / this->inverseMass;
    }

    Athena::Scalar RigidBody::getInverseMass() const
    {
        return this->inverseMass;
    }

    void RigidBody::setMass(const Athena::Scalar mass)
    {
        assert(mass != 0);
        this->inverseMass = ((Athena::Scalar)1.0) / mass;
    }

    void RigidBody::setInverseMass(const Athena::Scalar inverseMass)
    {
        this->inverseMass = inverseMass;
    }

    void RigidBody::setInertiaTensor(const Athena::Matrix3& inertiaTensor)
    {
        this->inverseInertiaTensor.setInverse(inertiaTensor);

    }

    Athena::Matrix3 RigidBody::getInertiaTensor() const
    {
        return this->inverseInertiaTensor.inverse();
    }

    Athena::Matrix3 RigidBody::getInertiaTensorWorld() const
    {
        return this->inverseInertiaTensorWorld.inverse();
    }

    void RigidBody::setInverseInertiaTensor(const Athena::Matrix3& inverseInertiaTensor)
    {
        this->inverseInertiaTensor = inverseInertiaTensor;
    }

    Athena::Matrix3 RigidBody::getInverseInertiaTensor() const
    {
        return this->inverseInertiaTensor;
    }

    Athena::Matrix3 RigidBody::getInverseInertiaTensorWorld() const
    {
        return this->inverseInertiaTensorWorld;
    }

    void RigidBody::setDamping(const Athena::Scalar linearDamping, const Athena::Scalar angularDamping)
    {
        this->linearDamping = linearDamping;
        this->angularDamping = angularDamping;
    }

    void RigidBody::setLinearDamping(const Athena::Scalar linearDamping)
    {
        this->linearDamping = linearDamping;
    }

    Athena::Scalar RigidBody::getLinearDamping() const
    {
        return this->linearDamping;
    }

    void RigidBody::setAngularDamping(const Athena::Scalar angularDamping)
    {
        this->angularDamping = angularDamping;
    }

    Athena::Scalar RigidBody::getAngularDamping() const
    {
        return this->angularDamping;
    }

    void RigidBody::setPosition(const Athena::Vector3& pos)
    {
        this->position = pos;
    }

    void RigidBody::setPosition(const Athena::Scalar x, const Athena::Scalar y, const Athena::Scalar z)
    {
        this->position.coordinates.x = x;
        this->position.coordinates.y = y;
        this->position.coordinates.z = z;
    }

    Athena::Vector3 RigidBody::getPosition() const
    {
        return this->position;
    }

    void RigidBody::setOrientation(const Athena::Quaternion& quat)
    {
        this->orientation = quat;
        orientation.normalize();
    }

    void RigidBody::setOrientation(const Athena::Scalar real, const Athena::Scalar x, const Athena::Scalar y, const Athena::Scalar z)
    {
        this->orientation.real = real;
        this->orientation.immaginary.coordinates.x = x;
        this->orientation.immaginary.coordinates.y = y;
        this->orientation.immaginary.coordinates.z = z;
        orientation.normalize();
    }

    void RigidBody::setOrientation(const Athena::Scalar real, const Athena::Vector3& imm)
    {
        this->orientation.real = real;
        this->orientation.immaginary = imm;
        orientation.normalize();
    }

    Athena::Quaternion RigidBody::getOrientation() const
    {
        return this->orientation;
    }

    void RigidBody::setVelocity(const Athena::Vector3& velocity)
    {
        this->velocity = velocity;
    }

    void RigidBody::setVelocity(const Athena::Scalar x, const Athena::Scalar y, const Athena::Scalar z)
    {
        this->velocity.coordinates.x = x;
        this->velocity.coordinates.y = y;
        this->velocity.coordinates.z = z;
    }

    Athena::Vector3 RigidBody::getVelocity() const
    {
        return this->velocity;
    }

    void RigidBody::setRotation(const Athena::Vector3& rotation)
    {
        this->rotation = rotation;
    }

    void RigidBody::setRotation(const Athena::Scalar x, const Athena::Scalar y, const Athena::Scalar z)
    {
        this->rotation.coordinates.x = x;
        this->rotation.coordinates.y = y;
        this->rotation.coordinates.z = z;
    }

    Athena::Vector3 RigidBody::getRotation() const
    {
        return this->rotation;
    }

    void RigidBody::setLastFrameAcceleration(const Athena::Vector3& linearAcceleration)
    {
        this->lastFrameAcceleration = linearAcceleration;
    }

    Athena::Vector3 RigidBody::getLastFrameAcceleration() const
    {
        return this->lastFrameAcceleration;
    }

    void RigidBody::calculateDerivedData()
    {
        //At each frame we calculate the transform matrix and transform the inverse inertia tensor into world coordinates
        this->orientation.normalize();

        //calculate the transform matrix
        calculateTransformMatrix(this->transformMatrix, this->position, this->orientation);

        //calculate the inertia tensor in world space
        transformInertiaTensor(this->inverseInertiaTensorWorld, this->orientation, this->inverseInertiaTensor, this->transformMatrix);
    }

    void RigidBody::addForce(const Athena::Vector3& force)
    {
        this->forceAccum += force;
    }

    void RigidBody::addVelocity(const Athena::Vector3& velocity)
    {
        this->velocity += velocity;
    }

    void RigidBody::addRotation(const Athena::Vector3& rotation)
    {
        this->rotation += rotation;
    }

    void RigidBody::addTorque(const Athena::Vector3& torque)
    {
        this->torqueAccum += torque;
    }

    void RigidBody::addForceAtBodyPoint(const Athena::Vector3& force, const Athena::Vector3& point)
    {
        //convert the point coordinates in world space
        Athena::Vector3 pt = getPointInWorldSpace(point);
        //the force should be in world space, while the point need to be transformed in local space
        addForceAtPoint(force, pt);
    }

    void RigidBody::addForceAtPoint(const Athena::Vector3& force, const Athena::Vector3& point)
    {
        //convert point to local space
        Athena::Vector3 pt = point;
        //here the conversion happens
        pt -= this->position;

        this->forceAccum += force;
        this->torqueAccum += Athena::Vector3::cross(pt, force);
        isAwake = true;
    }

    Athena::Vector3 RigidBody::getPointInWorldSpace(const Athena::Vector3& point)
    {
        Athena::Vector4 point4(point, 1);
        Athena::Vector4 result(this->transformMatrix * point4);
        return Athena::Vector3(result.coordinates.x, result.coordinates.y, result.coordinates.z);
    }

    void RigidBody::clearAccumulators()
    {
        this->forceAccum.clear();
        this->torqueAccum.clear();
    }

    void RigidBody::integrate(Athena::Scalar dt)
    {
        if(!isAwake)
            return;

        //calculate linear acceleration from force inputs
        this->lastFrameAcceleration = this->acceleration;
        this->lastFrameAcceleration.addScaledVector(this->forceAccum, this->inverseMass);

        //calculate angular accelaration from torque inputs
        Athena::Vector3 angularAcceleration = inverseInertiaTensorWorld * torqueAccum;

        //update linear velocity from both acceleration and impulse
        this->velocity.addScaledVector(this->lastFrameAcceleration, dt);

        //update angular velocity from both acceleration and impulse
        this->rotation.addScaledVector(angularAcceleration, dt);

        //impose drag
        this->velocity *= Athena::Math::scalarPow(linearDamping, dt);
        this->rotation *= Athena::Math::scalarPow(angularDamping, dt);
        //update linear position
        this->position.addScaledVector(this->velocity, dt);

        //update angular position
        this->orientation.addScaledVector(this->rotation, dt);

        //update the transform matrix and the inertia tensor matrix, based on the newly calculated
        //position and rotation
        calculateDerivedData();
        clearAccumulators();

        if(canSleep)
        {
            Athena::Scalar currentMotion = Athena::Vector3::dot(this->velocity, this->velocity) +
                    Athena::Vector3::dot(this->rotation, this->rotation);
                
            Athena::Scalar bias = Athena::Math::scalarPow(0.5, dt);
            this->motion = bias * this->motion + (1 - bias) * currentMotion;
            
            if(motion < sleepEpsilon)
            {
                setAwake(false);
            }
            else if(motion > 10 * sleepEpsilon)
                motion = 10 * sleepEpsilon;
        }
    }

    void RigidBody::setAwake(const bool awake)
    {
        if(awake)
        {
            isAwake = true;
            motion = sleepEpsilon * 2.0f;
        }
        else
        {
            isAwake = false;
            velocity.clear();
            rotation.clear();
        }
    }

    bool RigidBody::getAwake()
    {
        return this->isAwake;
    }

    void RigidBody::setCanSleep(const bool canSleep)
    {
        this->canSleep = canSleep;
        if (!canSleep && !isAwake) setAwake();
    }

    bool RigidBody::getCanSleep()
    {
        return this->canSleep;
    }
}