#include <Quaternion.hpp>

namespace Athena {
    Quaternion::Quaternion() {
        this->immaginary = Vector3(0, 0, 0);
        this->real = 1.0;
    }

    Quaternion::Quaternion(const Scalar& a, const Scalar& b, const Scalar& c, const Scalar& d) {
        this->immaginary = Vector3(a, b, c);
        this->real = d;
    }

    Quaternion::Quaternion(const Vector3& immaginary, const Scalar& real) {
        this->immaginary = Vector3(immaginary);
        this->real = real;
    }

    Quaternion::Quaternion(const Vector4& quaternion) {
        this->immaginary = Vector3(quaternion.coordinates.x, quaternion.coordinates.y, quaternion.coordinates.z);
        this->real = quaternion.coordinates.w;
    }

    Quaternion::Quaternion(const Quaternion& quaternion) {
        this->immaginary = quaternion.getImmaginaryPart();
        this->real = quaternion.getRealPart();
    }

    Vector3 Quaternion::getImmaginaryPart() const {
        return this->immaginary;
    }

    Scalar Quaternion::getRealPart() const {
        return this->real;
    }

    Quaternion Quaternion::fromMatrix(const Matrix3& matrix) const {
        return Quaternion::Matrix3ToQuaternion(matrix);
    }

    Quaternion Quaternion::fromEulerAngles(const Vector3& eulerAngles) const {
        return Quaternion::EulerAnglesToQuaternion(eulerAngles);
    }

    Quaternion Quaternion::fromAxisAngle(const Degree& angle, const Vector3& axis) const {
        return Quaternion::AxisAngleToQuaternion(angle, axis);
    }

    Matrix3 Quaternion::toMatrix3() const {
        return Quaternion::QuaternionToMatrx3(*this);
    }

    Vector3 Quaternion::toEulerAngles() const {
        return Quaternion::QuaternionToEulerAngles(*this);
    }

    void Quaternion::toAxisAngle(Degree& angle, Vector3& axis) const {
        Quaternion::QuaternionToAxisAngle(*this, angle, axis);
    }

    void Quaternion::conjugate() {
        this->immaginary = -this->immaginary;
    }

    Quaternion Quaternion::conjugated() const {
        return Quaternion(-this->immaginary, this->real);
    }

    Scalar Quaternion::squareMagnitude() const {
        return this->immaginary.squareMagnitude() + this->real * this->real;
    }

    Scalar Quaternion::magnitude() const {
        return std::sqrt(this->squareMagnitude());
    }

    void Quaternion::invert() {
        this->conjugate();
        *this /= this->squareMagnitude();
    }

    Quaternion Quaternion::inverse() const {
        return this->conjugated() / this->squareMagnitude();
    }

    Quaternion Quaternion::operator /(const Scalar& k) const {
        return Quaternion(this->immaginary / k, this->real / k);
    }

    Quaternion Quaternion::operator *(const Scalar& k) const {
        return Quaternion(this->immaginary * k, this->real * k);
    }

    void Quaternion::operator /=(const Scalar& k) {
        this->immaginary /= k;
        this->real /= k;
    }

    void Quaternion::operator *=(const Scalar& k) {
        this->immaginary *= k;
        this->real *= k;
    }

    void Quaternion::operator +=(const Quaternion& quaternion) {
        this->immaginary += quaternion.getImmaginaryPart();
        this->real += quaternion.getRealPart();
    }

    void Quaternion::operator -=(const Quaternion& quaternion) {
        this->immaginary -= quaternion.getImmaginaryPart();
        this->real -= quaternion.getRealPart();
    }

    bool Quaternion::operator ==(const Quaternion& quaternion) const {
        return (this->immaginary == quaternion.getImmaginaryPart() && this->real == quaternion.getRealPart());
    }

    bool Quaternion::operator !=(const Quaternion& quaternion) const {
        return !((*this) == quaternion);
    }

    Vector3 Quaternion::rotateVectorByThisQuaternion(const Vector3& vectorToRotate) {
        Quaternion result = this->conjugated() * Quaternion(vectorToRotate, 0.0) * (*this);

        if (result.getRealPart() != 0)
            throw std::invalid_argument("ERROR::in QUATERNION rotateVectorByThisQuaternion function, the real part is not 0!");

        return result.getImmaginaryPart();
    }

    Vector4* Quaternion::asVector4() const {
        return Quaternion::AsVector4(*this);
    }

    void Quaternion::print() const {
        std::cout << "( " << this->immaginary.coordinates.x << ", " << this->immaginary.coordinates.y << ", " 
            << this->immaginary.coordinates.z << ", " << this->real << " )" << std::endl;
    }

    Quaternion operator *(const Quaternion& a, const Quaternion& b) {
        return Quaternion(
            a.getRealPart() * b.getImmaginaryPart() + a.getImmaginaryPart() * b.getRealPart() + Vector3::cross(a.getImmaginaryPart(), b.getImmaginaryPart()),  // imaginary part
            a.getRealPart() * b.getRealPart() - Vector3::dot(a.getImmaginaryPart(), b.getImmaginaryPart()) // real part
        );
    }
}