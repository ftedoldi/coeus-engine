#include <Transform.hpp>

namespace Odysseus {

    Transform::Transform(const Athena::Vector3& pos, const Athena::Quaternion& rot, const Athena::Vector3& scale) 
    : rotation(_rotation), worldToLocalMatrix(_worldToLocalMatrix), localToWorldMatrix(_localToWorldMatrix), _position(position) {
        this->position = pos;
        this->_rotation = rot;
        this->localScale = scale;

        this->_childrenTree = new Zeus::Tree(this);

        this->_localToWorldMatrix = Athena::Matrix4(
            Athena::Vector4(1, 0, 0, _position.coordinates.x), 
            Athena::Vector4(0, 1, 0, _position.coordinates.y),
            Athena::Vector4(0, 0, 1, _position.coordinates.z), 
            Athena::Vector4(0, 0, 0, 1)
        );

        this->_worldToLocalMatrix = Athena::Matrix4(
            Athena::Vector4(1, 0, 0, -_position.coordinates.x), 
            Athena::Vector4(0, 1, 0, -_position.coordinates.y),
            Athena::Vector4(0, 0, 1, -_position.coordinates.z), 
            Athena::Vector4(0, 0, 0, 1)
        );

        this->name = "Scene Object";
    }

    Transform::Transform(const Athena::Vector3& pos, const Athena::Vector3& eulerAnglesRotation, const Athena::Vector3& scale) 
    : rotation(_rotation), worldToLocalMatrix(_worldToLocalMatrix), localToWorldMatrix(_localToWorldMatrix), _position(position) {
        this->position = pos;
        this->_rotation = Athena::Quaternion::EulerAnglesToQuaternion(eulerAnglesRotation);
        this->localScale = scale;

        this->_childrenTree = new Zeus::Tree(this);

        this->_localToWorldMatrix = Athena::Matrix4(
            Athena::Vector4(1, 0, 0, _position.coordinates.x), 
            Athena::Vector4(0, 1, 0, _position.coordinates.y),
            Athena::Vector4(0, 0, 1, _position.coordinates.z), 
            Athena::Vector4(0, 0, 0, 1)
        );

        this->_worldToLocalMatrix = Athena::Matrix4(
            Athena::Vector4(1, 0, 0, -_position.coordinates.x), 
            Athena::Vector4(0, 1, 0, -_position.coordinates.y),
            Athena::Vector4(0, 0, 1, -_position.coordinates.z), 
            Athena::Vector4(0, 0, 0, 1)
        );

        this->name = "Scene Object";
    }

    Transform::Transform(const Athena::Vector3& pos, const Athena::Vector3& eulerAnglesRotation) 
    : rotation(_rotation), worldToLocalMatrix(_worldToLocalMatrix), localToWorldMatrix(_localToWorldMatrix), _position(position)  {
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

        this->_worldToLocalMatrix = Athena::Matrix4(
            Athena::Vector4(1, 0, 0, -_position.coordinates.x), 
            Athena::Vector4(0, 1, 0, -_position.coordinates.y),
            Athena::Vector4(0, 0, 1, -_position.coordinates.z), 
            Athena::Vector4(0, 0, 0, 1)
        );

        this->name = "Scene Object";
    }

    Transform::Transform(const Athena::Vector3& pos) 
    : rotation(_rotation), worldToLocalMatrix(_worldToLocalMatrix), localToWorldMatrix(_localToWorldMatrix), _position(position) {
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

        this->_worldToLocalMatrix = Athena::Matrix4(
            Athena::Vector4(1, 0, 0, -_position.coordinates.x), 
            Athena::Vector4(0, 1, 0, -_position.coordinates.y),
            Athena::Vector4(0, 0, 1, -_position.coordinates.z), 
            Athena::Vector4(0, 0, 0, 1)
        );

        this->name = "Scene Object";
    }

    Transform::Transform(const Athena::Quaternion& rot) 
    : rotation(_rotation), worldToLocalMatrix(_worldToLocalMatrix), localToWorldMatrix(_localToWorldMatrix), _position(position) {
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

        this->_worldToLocalMatrix = Athena::Matrix4(
            Athena::Vector4(1, 0, 0, -_position.coordinates.x), 
            Athena::Vector4(0, 1, 0, -_position.coordinates.y),
            Athena::Vector4(0, 0, 1, -_position.coordinates.z), 
            Athena::Vector4(0, 0, 0, 1)
        );

        this->name = "Scene Object";
    }

    Transform::Transform(const Transform& t) 
    : rotation(_rotation), worldToLocalMatrix(_worldToLocalMatrix), localToWorldMatrix(_localToWorldMatrix), _position(position) {
        this->position = t.position;
        this->_rotation = t.rotation;
        this->localScale = t.localScale;

        this->_childrenTree = t.childrenTree();

        this->_localToWorldMatrix = t.localToWorldMatrix;

        this->_worldToLocalMatrix = t.worldToLocalMatrix;

        this->name = t.name;
    }

    Transform::Transform() 
    : rotation(_rotation), worldToLocalMatrix(_worldToLocalMatrix), localToWorldMatrix(_localToWorldMatrix), _position(position) {
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

        this->_worldToLocalMatrix = Athena::Matrix4(
            Athena::Vector4(1, 0, 0, -_position.coordinates.x), 
            Athena::Vector4(0, 1, 0, -_position.coordinates.y),
            Athena::Vector4(0, 0, 1, -_position.coordinates.z), 
            Athena::Vector4(0, 0, 0, 1)
        );

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

    Athena::Versor2 Transform::transformDirection(const Athena::Versor2& versor) const
    {
        auto tmp = localToWorldMatrix * Athena::Vector4(versor.asVector2(), 0, 1);
        return Athena::Versor2(tmp.coordinates.x, tmp.coordinates.y);
    }

    Athena::Versor3 Transform::transformDirection(const Athena::Versor3& versor) const
    {
        auto tmp = localToWorldMatrix * Athena::Vector4(versor.asVector3(), 1);
        return Athena::Versor3(tmp.coordinates.x, tmp.coordinates.y, tmp.coordinates.z);
    }

    Athena::Versor4 Transform::transformDirection(const Athena::Versor4& versor) const
    {
        if (versor.coordinates.w != 1.f)
            throw std::invalid_argument("INVALID_VERSOR::in order to transform a Versor4 to World Coordinates the last component might be 1");

        return Athena::Versor4(localToWorldMatrix * Athena::Vector4(versor.asVector4()));
    }

    Athena::Vector2 Transform::transformVector(const Athena::Vector2& vector) const
    {
        auto tmp = localToWorldMatrix * Athena::Vector4(vector, 0, 1);
        return Athena::Vector2(tmp.coordinates.x, tmp.coordinates.y);
    }

    Athena::Vector3 Transform::transformVector(const Athena::Vector3& vector) const
    {
        auto tmp = localToWorldMatrix * Athena::Vector4(vector, 1);
        return Athena::Vector3(tmp.coordinates.x, tmp.coordinates.y, tmp.coordinates.z);
    }

    Athena::Vector4 Transform::transformVector(const Athena::Vector4& vector) const
    {
        if (vector.coordinates.w != 1.f)
            throw std::invalid_argument("INVALID_VECTOR::in order to transform a Vector4 to World Coordinates the last component might be 1");

        return localToWorldMatrix * Athena::Vector4(vector);
    }

    Athena::Point2 Transform::transformPoint(const Athena::Point2& point) const
    {
        auto tmp = localToWorldMatrix * Athena::Vector4(point.asVector2(), 0, 1);
        return Athena::Point2(tmp.coordinates.x, tmp.coordinates.y);
    }

    Athena::Point3 Transform::transformPoint(const Athena::Point3& point) const
    {
        auto tmp = localToWorldMatrix * Athena::Vector4(point.asVector3(), 1);
        return Athena::Point3(tmp.coordinates.x, tmp.coordinates.y, tmp.coordinates.z);
    }

    Athena::Point4 Transform::transformPoint(const Athena::Point4& point) const
    {
        if (point.coordinates.w != 1.f)
            throw std::invalid_argument("INVALID_POINT::in order to transform a Point4 to World Coordinates the last component might be 1");

        return localToWorldMatrix * Athena::Vector4(point.asVector4());
    }

    Athena::Versor2 Transform::inverseTransformDirection(const Athena::Versor2& versor) const
    {
        auto tmp = worldToLocalMatrix * Athena::Vector4(versor.asVector2(), 0, 1);
        return Athena::Versor2(tmp.coordinates.x, tmp.coordinates.y);
    }

    Athena::Versor3 Transform::inverseTransformDirection(const Athena::Versor3& versor) const
    {
        auto tmp = worldToLocalMatrix * Athena::Vector4(versor.asVector3(), 1);
        return Athena::Versor3(tmp.coordinates.x, tmp.coordinates.y, tmp.coordinates.z);
    }

    Athena::Versor4 Transform::inverseTransformDirection(const Athena::Versor4& versor) const
    {
        if (versor.coordinates.w != 1.f)
            throw std::invalid_argument("INVALID_VERSOR::in order to transform a Versor4 to Local Coordinates the last component might be 1");

        return Athena::Versor4(worldToLocalMatrix * Athena::Vector4(versor.asVector4()));
    }

    Athena::Vector2 Transform::inverseTransformVector(const Athena::Vector2& vector) const
    {
        auto tmp = worldToLocalMatrix * Athena::Vector4(vector, 0, 1);
        return Athena::Vector2(tmp.coordinates.x, tmp.coordinates.y);
    }

    Athena::Vector3 Transform::inverseTransformVector(const Athena::Vector3& vector) const
    {
        auto tmp = worldToLocalMatrix * Athena::Vector4(vector, 1);
        return Athena::Vector3(tmp.coordinates.x, tmp.coordinates.y, tmp.coordinates.z);
    }

    Athena::Vector4 Transform::inverseTransformVector(const Athena::Vector4& vector) const
    {
        if (vector.coordinates.w != 1.f)
            throw std::invalid_argument("INVALID_VECTOR::in order to transform a Vector4 to Local Coordinates the last component might be 1");

        return worldToLocalMatrix * Athena::Vector4(vector);
    }

    Athena::Point2 Transform::inverseTransformPoint(const Athena::Point2& point) const
    {
        auto tmp = worldToLocalMatrix * Athena::Vector4(point.asVector2(), 0, 1);
        return Athena::Point2(tmp.coordinates.x, tmp.coordinates.y);
    }

    Athena::Point3 Transform::inverseTransformPoint(const Athena::Point3& point) const
    {
        auto tmp = worldToLocalMatrix * Athena::Vector4(point.asVector3(), 1);
        return Athena::Point3(tmp.coordinates.x, tmp.coordinates.y, tmp.coordinates.z);
    }

    Athena::Point4 Transform::inverseTransformPoint(const Athena::Point4& point) const
    {
        if (point.coordinates.w != 1.f)
            throw std::invalid_argument("INVALID_POINT::in order to transform a Point4 to Local Coordinates the last component might be 1");

        return worldToLocalMatrix * Athena::Vector4(point.asVector4());
    }

    void Transform::addChild(Transform& child)
    {
        this->_childrenTree->addChild(child);
    }

    void Transform::setFather(Transform& father)
    {
        Zeus::Node* node = new Zeus::Node;
        node->father = nullptr;
        node->transform = &father;
        node->father->children.push_back(this->_childrenTree->root);

        this->_childrenTree->root->father = node;
    }

    Zeus::Node* Transform::getChild(const int& index)
    {
        return this->_childrenTree->getChild(index);
    }

    Zeus::Node* Transform::getChild(const std::string& name)
    {
        return this->_childrenTree->getChild(name);
    }

    Transform* Transform::getChildTransform(const int& index) const
    {
        return this->_childrenTree->getChild(index)->transform;
    }

    Transform* Transform::getChildTransform(const std::string& name) const
    {
        return this->_childrenTree->getChild(name)->transform;
    }

    bool Transform::operator == (const Transform& b) const
    {
        return (this->position == b.position && this->rotation == b.rotation && this->localScale == b.localScale);
    }

    bool Transform::operator != (const Transform& b) const
    {
        return !(*(this) == b);
    }

    Transform Transform::operator * (const Transform& b) const
    {
        Athena::Vector3 newScale = Athena::Vector3(
            this->localScale.coordinates.x * b.localScale.coordinates.x,
            this->localScale.coordinates.y * b.localScale.coordinates.y,
            this->localScale.coordinates.z * b.localScale.coordinates.z
        );

        Athena::Quaternion newRotation = b.rotation * this->rotation;

        Athena::Vector3 newPosition = this->position + this->rotation.rotateVectorByThisQuaternion(Athena::Vector3(
            this->localScale.coordinates.x * b.position.coordinates.x,
            this->localScale.coordinates.y * b.position.coordinates.y,
            this->localScale.coordinates.z * b.position.coordinates.z
        ));

        return Transform(newPosition, newRotation, newScale);
    }

    Transform* Transform::inverse() const
    {
        Athena::Vector3 newScale = Athena::Vector3(
            1/this->localScale.coordinates.x,
            1/this->localScale.coordinates.y,
            1/this->localScale.coordinates.z
        );

        Athena::Quaternion newRotation = this->rotation.inverse();

        Athena::Vector3 newPosition = newRotation.rotateVectorByThisQuaternion(Athena::Vector3(
            -this->position.coordinates.x * newScale.coordinates.x,
            -this->position.coordinates.y * newScale.coordinates.y,
            -this->position.coordinates.z * newScale.coordinates.z
        ));

        return new Transform(newPosition, newRotation, newScale);
    }

    Transform::~Transform()
    {
        this->_childrenTree->deleteTree();
        delete this->_childrenTree;
    }

    Transform* compositeTransformBetween (Transform* a, Transform* b)
    {
        return new Transform(
            a->position + a->rotation.rotateVectorByThisQuaternion(
                Athena::Vector3(
                    a->localScale.coordinates.x * b->position.coordinates.x,
                    a->localScale.coordinates.y * b->position.coordinates.y,
                    a->localScale.coordinates.z * b->position.coordinates.z
                )
            ), 
            a->rotation * b->rotation,
            Athena::Vector3(
                a->localScale.coordinates.x * b->localScale.coordinates.x,
                a->localScale.coordinates.y * b->localScale.coordinates.y,
                a->localScale.coordinates.z * b->localScale.coordinates.z
            )
        );
    }

}