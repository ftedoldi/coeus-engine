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

    Matrix3 Quaternion::toMatrix3() const {
        return Matrix3();
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
        return Quaternion();
    }

    Quaternion Quaternion::operator *(const Scalar& k) const {
        return Quaternion();
    }

    Quaternion Quaternion::operator /=(const Scalar& k) {
        return Quaternion();
    }

    Quaternion Quaternion::operator *=(const Scalar& k) {
        return Quaternion();
    }

    Quaternion Quaternion::operator +=(const Quaternion& quaternion) {
        return Quaternion();
    }

    Quaternion Quaternion::operator -=(const Quaternion& quaternion) {
        return Quaternion();
    }

    Vector3 Quaternion::rotateVectorByThisQuaternion(const Vector3& vectorToRotate) {
        Quaternion result = this->conjugated() * Quaternion(vectorToRotate, 0.0) * (*this);

        if (result.getRealPart() != 0)
            throw std::invalid_argument("ERROR::in QUATERNION rotateVectorByThisQuaternion function, the real part is not 0!");

        return result.getImmaginaryPart();
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