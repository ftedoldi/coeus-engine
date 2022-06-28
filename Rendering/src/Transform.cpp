#include <Transform.hpp>

namespace Odysseus {

    Transform::Transform(const Athena::Vector3& pos, const Athena::Quaternion& rot, const Athena::Vector3& scale) 
    : rotation(_rotation), worldToLocalMatrix(_worldToLocalMatrix), localToWorldMatrix(_localToWorldMatrix) {
        this->position = pos;
        this->_rotation = rot;
        this->localScale = scale;

        this->_childrenTree = new Zeus::Tree(this);

        this->_localToWorldMatrix = Athena::Matrix4(
            Athena::Vector4(1, 0, 0, position.coordinates.x), 
            Athena::Vector4(0, 1, 0, position.coordinates.y),
            Athena::Vector4(0, 0, 1, position.coordinates.z), 
            Athena::Vector4(0, 0, 0, 1)
        );

        this->_worldToLocalMatrix = this->_localToWorldMatrix.inverse();

        this->name = "Scene Object";
    }

    Transform::Transform(const Athena::Vector3& pos, const Athena::Vector3& eulerAnglesRotation, const Athena::Vector3& scale) 
    : rotation(_rotation), worldToLocalMatrix(_worldToLocalMatrix), localToWorldMatrix(_localToWorldMatrix) {
        this->position = pos;
        this->_rotation = Athena::Quaternion::EulerAnglesToQuaternion(eulerAnglesRotation);
        this->localScale = scale;

        this->_childrenTree = new Zeus::Tree(this);

        this->_localToWorldMatrix = Athena::Matrix4(
            Athena::Vector4(1, 0, 0, position.coordinates.x), 
            Athena::Vector4(0, 1, 0, position.coordinates.y),
            Athena::Vector4(0, 0, 1, position.coordinates.z), 
            Athena::Vector4(0, 0, 0, 1)
        );

        this->_worldToLocalMatrix = this->_localToWorldMatrix.inverse();

        this->name = "Scene Object";
    }

    Transform::Transform(const Athena::Vector3& pos, const Athena::Vector3& eulerAnglesRotation) 
    : rotation(_rotation), worldToLocalMatrix(_worldToLocalMatrix), localToWorldMatrix(_localToWorldMatrix)  {
        this->position = pos;
        this->_rotation = Athena::Quaternion::EulerAnglesToQuaternion(eulerAnglesRotation);
        this->localScale = Athena::Vector3(1, 1, 1);

        this->_childrenTree = new Zeus::Tree(this);

        this->_localToWorldMatrix = Athena::Matrix4(
            Athena::Vector4(1, 0, 0, position.coordinates.x), 
            Athena::Vector4(0, 1, 0, position.coordinates.y),
            Athena::Vector4(0, 0, 1, position.coordinates.z), 
            Athena::Vector4(0, 0, 0, 1)
        );

        this->_worldToLocalMatrix = this->_localToWorldMatrix.inverse();

        this->name = "Scene Object";
    }

    Transform::Transform(const Athena::Vector3& pos) 
    : rotation(_rotation), worldToLocalMatrix(_worldToLocalMatrix), localToWorldMatrix(_localToWorldMatrix) {
        this->position = pos;
        this->_rotation = Athena::Quaternion::Identity();
        this->localScale = Athena::Vector3(1, 1, 1);

        this->_childrenTree = new Zeus::Tree(this);

        this->_localToWorldMatrix = Athena::Matrix4(
            Athena::Vector4(1, 0, 0, position.coordinates.x), 
            Athena::Vector4(0, 1, 0, position.coordinates.y),
            Athena::Vector4(0, 0, 1, position.coordinates.z), 
            Athena::Vector4(0, 0, 0, 1)
        );

        this->_worldToLocalMatrix = this->_localToWorldMatrix.inverse();

        this->name = "Scene Object";
    }

    Transform::Transform(const Athena::Quaternion& rot) 
    : rotation(_rotation), worldToLocalMatrix(_worldToLocalMatrix), localToWorldMatrix(_localToWorldMatrix) {
        this->position = Athena::Vector3(0, 0, 0);
        this->_rotation = rot;
        this->localScale = Athena::Vector3(1, 1, 1);

        this->_childrenTree = new Zeus::Tree(this);

        this->_localToWorldMatrix = Athena::Matrix4(
            Athena::Vector4(1, 0, 0, position.coordinates.x), 
            Athena::Vector4(0, 1, 0, position.coordinates.y),
            Athena::Vector4(0, 0, 1, position.coordinates.z), 
            Athena::Vector4(0, 0, 0, 1)
        );

        this->_worldToLocalMatrix = this->_localToWorldMatrix.inverse();

        this->name = "Scene Object";
    }

    Transform::Transform(const Transform& t) 
    : rotation(_rotation), worldToLocalMatrix(_worldToLocalMatrix), localToWorldMatrix(_localToWorldMatrix) {
        this->position = t.position;
        this->_rotation = t.rotation;
        this->localScale = t.localScale;

        this->_childrenTree = t.childrenTree();

        this->_localToWorldMatrix = t.localToWorldMatrix;

        this->_localToWorldMatrix.print();

        this->_worldToLocalMatrix = t.worldToLocalMatrix;

        this->name = t.name;
    }

    Transform::Transform() 
    : rotation(_rotation), worldToLocalMatrix(_worldToLocalMatrix), localToWorldMatrix(_localToWorldMatrix) {
        this->position = Athena::Vector3(0, 0, 0);
        this->_rotation = Athena::Quaternion::Identity();
        this->localScale = Athena::Vector3(1, 1, 1);

        this->_childrenTree = new Zeus::Tree(this);

        this->_localToWorldMatrix = Athena::Matrix4(
            Athena::Vector4(1, 0, 0, this->position.coordinates.x), 
            Athena::Vector4(0, 1, 0, this->position.coordinates.y),
            Athena::Vector4(0, 0, 1, this->position.coordinates.z), 
            Athena::Vector4(0, 0, 0, 1)
        );

        this->_localToWorldMatrix.print();

        this->_worldToLocalMatrix = this->_localToWorldMatrix.inverse();

        this->name = "Scene Object";
    }

    Zeus::Tree* Transform::childrenTree() const {
        return new Zeus::Tree(this->_childrenTree);
    }

    Athena::Vector3 Transform::up() const
    {
        return this->_rotation.rotateVectorByThisQuaternion(Athena::Vector3::up());
    }

    Athena::Vector3 Transform::forward() const
    {
        return this->_rotation.rotateVectorByThisQuaternion(Athena::Vector3::forward());
    }

    Athena::Vector3 Transform::right() const
    {
        return this->_rotation.rotateVectorByThisQuaternion(Athena::Vector3::right());
    }

    Transform Transform::translate(const Athena::Vector3& destination) const
    {
        return Transform(this->position + destination, this->_rotation, this->localScale);
    }

    Transform Transform::nonUniformScaleBy(const Athena::Vector3& scale) const
    {
        return Transform(
            this->position, 
            this->_rotation,
            Athena::Vector3(
                this->localScale.coordinates.x * scale.coordinates.x, 
                this->localScale.coordinates.y * scale.coordinates.y, 
                this->localScale.coordinates.z * scale.coordinates.z
            )
        );
    }

    Transform Transform::uniformScaleBy(const Athena::Scalar& uniformScale) const
    {
        return Transform(this->position, this->_rotation, this->localScale * uniformScale);
    }

    Transform Transform::rotateAroundAxis(const Athena::Vector3& axis, const Athena::Scalar& angle) const
    {
        return Transform(this->position, this->_rotation.fromAxisAngle(angle, axis), this->localScale);
    }

    Transform Transform::rotateAroundAxis(const Athena::Vector4& axisAngle) const
    {
        return Transform(
            this->position, 
            this->_rotation.fromAxisAngle(
                axisAngle.coordinates.w, 
                Athena::Vector3(
                    axisAngle.coordinates.x, 
                    axisAngle.coordinates.y, 
                    axisAngle.coordinates.z
                )
            ), 
            this->localScale
        );
    }

    Transform Transform::rotateOfEulerAngles(const Athena::Vector3 eulerAngles) const
    {
        return Transform(this->position, this->_rotation.fromEulerAngles(eulerAngles), this->localScale);
    }

    Transform Transform::rotateOfMatrix3(const Athena::Matrix3 matrix) const
    {
        return Transform(this->position, this->_rotation.fromMatrix(matrix), this->localScale);
    }

    Transform Transform::rotate(const Athena::Quaternion& rotationQuaternion) const
    {
        return Transform(this->position, rotationQuaternion, this->localScale);
    }

    // TODO: Test this function
    Transform Transform::lookAt(const Athena::Vector3& position) const
    {
        Athena::Vector3 direction = (position - this->position).normalized();
        Athena::Quaternion newRotation = Athena::Quaternion::RotationBetweenVectors(forward(), direction);

        Athena::Vector3 r = Athena::Vector3::cross(direction, up());
        Athena::Vector3 desiredUp = Athena::Vector3::cross(r, direction);

        Athena::Vector3 newUp = newRotation.rotateVectorByThisQuaternion(up());
        Athena::Quaternion finalRotation = Athena::Quaternion::RotationBetweenVectors(newUp, desiredUp);

        return Transform(this->position, finalRotation, this->localScale);
    }

    // TODO: Test this function
    Transform Transform::lookAt(const Transform& target) const
    {
        Athena::Vector3 direction = (target.position - this->position).normalized();
        Athena::Quaternion newRotation = Athena::Quaternion::RotationBetweenVectors(forward(), direction);

        Athena::Vector3 r = Athena::Vector3::cross(direction, up());
        Athena::Vector3 desiredUp = Athena::Vector3::cross(r, direction);

        Athena::Vector3 newUp = newRotation.rotateVectorByThisQuaternion(up());
        Athena::Quaternion finalRotation = Athena::Quaternion::RotationBetweenVectors(newUp, desiredUp);

        return Transform(this->position, finalRotation, this->localScale);
    }

    void Transform::translate(const Athena::Vector3& destination)
    {
        this->position += destination;
    }

    void Transform::nonUniformScaleBy(const Athena::Vector3& scale)
    {
        this->localScale.coordinates.x *= scale.coordinates.x;
        this->localScale.coordinates.y *= scale.coordinates.y;
        this->localScale.coordinates.z *= scale.coordinates.z;
    }

    void Transform::uniformScaleBy(const Athena::Scalar& uniformScale)
    {
        this->localScale *= uniformScale;
    }

    void Transform::rotateAroundAxis(const Athena::Vector3& axis, const Athena::Scalar& angle)
    {
        this->_rotation = this->_rotation.fromAxisAngle(angle, axis);
    }

    void Transform::rotateOfEulerAngles(const Athena::Vector3 eulerAngles)
    {
        this->_rotation = this->_rotation.fromEulerAngles(eulerAngles);
    }

    void Transform::rotateAroundAxis(const Athena::Vector4& axisAngle)
    {
        this->_rotation = this->_rotation.fromAxisAngle(
            axisAngle.coordinates.w, 
            Athena::Vector3(
                axisAngle.coordinates.x, 
                axisAngle.coordinates.y, 
                axisAngle.coordinates.z
            )
        );
    }

    void Transform::rotateOfMatrix3(const Athena::Matrix3 matrix)
    {
        this->_rotation = this->_rotation.fromMatrix(matrix);
    }

    void Transform::rotate(const Athena::Quaternion& rotationQuaternion)
    {
        this->_rotation = rotationQuaternion;
    }

    void Transform::lookAt(const Athena::Vector3& position)
    {
        Athena::Vector3 direction = (position - this->position).normalized();
        Athena::Quaternion newRotation = Athena::Quaternion::RotationBetweenVectors(forward(), direction);

        Athena::Vector3 r = Athena::Vector3::cross(direction, up());
        Athena::Vector3 desiredUp = Athena::Vector3::cross(r, direction);

        Athena::Vector3 newUp = newRotation.rotateVectorByThisQuaternion(up());
        Athena::Quaternion finalRotation = Athena::Quaternion::RotationBetweenVectors(newUp, desiredUp);

        this->_rotation = finalRotation;
    }

    void Transform::lookAt(const Transform& target)
    {
        Athena::Vector3 direction = (target.position - this->position).normalized();
        Athena::Quaternion newRotation = Athena::Quaternion::RotationBetweenVectors(forward(), direction);

        Athena::Vector3 r = Athena::Vector3::cross(direction, up());
        Athena::Vector3 desiredUp = Athena::Vector3::cross(r, direction);

        Athena::Vector3 newUp = newRotation.rotateVectorByThisQuaternion(up());
        Athena::Quaternion finalRotation = Athena::Quaternion::RotationBetweenVectors(newUp, desiredUp);

        this->_rotation = finalRotation;
    }

    Transform Transform::transformDirection(const Athena::Versor2& versor) const
    {
        return Transform();
    }

    Transform Transform::transformVector(const Athena::Vector2& vector) const
    {
        return Transform();
    }

    Transform Transform::transformVector(const Athena::Vector3& vector) const
    {
        return Transform();
    }

    Transform Transform::transformVector(const Athena::Vector4& vector) const
    {
        return Transform();
    }

    Transform Transform::transformPoint(const Athena::Point2& point) const
    {
        return Transform();
    }

    Transform Transform::inverseTransformDirection(const Athena::Versor2& versor) const
    {
        return Transform();
    }

    Transform Transform::inverseTransformVector(const Athena::Vector2& vector) const
    {
        return Transform();
    }

    Transform Transform::inverseTransformVector(const Athena::Vector3& vector) const
    {
        return Transform();
    }

    Transform Transform::inverseTransformVector(const Athena::Vector4& vector) const
    {
        return Transform();
    }

    Transform Transform::inverseTransformPoint(const Athena::Point2& point) const
    {
        return Transform();
    }

    void Transform::addChild(const Transform& child)
    {

    }

    Zeus::Node* Transform::getChild(const int& index)
    {
        return nullptr;
    }

    Zeus::Node* Transform::getChild(const std::string& name)
    {
        return nullptr;
    }

    Transform Transform::getChildTransform(const int& index) const
    {
        return Transform();
    }

    Transform Transform::getChildTransform(const std::string& name) const
    {
        return Transform();
    }

    bool Transform::operator == (const Transform& b) const
    {
        return false;
    }

    bool Transform::operator != (const Transform& b) const
    {
        return false;
    }

    Transform Transform::operator * (const Transform& b) const
    {
        return Transform();
    }

    Transform Transform::inverse() const
    {
        return Transform();
    }

    Transform::~Transform()
    {
        //delete this->_childrenTree;
    }

}