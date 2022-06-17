#include <Vector4.hpp>

namespace Athena {
    Vector4::Vector4() {
        coordinates = Vector4Coordinates<Scalar>();
        this->coordinates.x = (Scalar) 0.0;
        this->coordinates.y = (Scalar) 0.0;
        this->coordinates.z = (Scalar) 0.0;
        this->coordinates.w = (Scalar) 0.0;
    }

    Vector4::Vector4(const Scalar& x, const Scalar& y, const Scalar& z, const Scalar& w) {
        coordinates = Vector4Coordinates<Scalar>();
        this->coordinates.x = x;
        this->coordinates.y = y;
        this->coordinates.z = z;
        this->coordinates.w = w;
    }

    Vector4::Vector4(const Vector2& vector, const Scalar& z, const Scalar& w) {
        coordinates = Vector4Coordinates<Scalar>();
        this->coordinates.x = vector.coordinates.x;
        this->coordinates.y = vector.coordinates.y;
        this->coordinates.z = z;
        this->coordinates.w = w;
    }

    Vector4::Vector4(const Vector3& vector, const Scalar& w) {
        coordinates = Vector4Coordinates<Scalar>();
        this->coordinates.x = vector.coordinates.x;
        this->coordinates.y = vector.coordinates.y;
        this->coordinates.z = vector.coordinates.z;
        this->coordinates.w = w;
    }
    
    Vector4::Vector4(const Vector4& vector) {
        coordinates = Vector4Coordinates<Scalar>();
        this->coordinates.x = vector.coordinates.x;
        this->coordinates.y = vector.coordinates.y;
        this->coordinates.z = vector.coordinates.z;
        this->coordinates.w = vector.coordinates.w;
    }

    Scalar Vector4::dot(const Vector4& vector) const {
        return this->coordinates.x * vector.coordinates.x +
                this->coordinates.y * vector.coordinates.y +
                this->coordinates.z * vector.coordinates.z +
                this->coordinates.w * vector.coordinates.w;
    }

    Scalar Vector4::magnitude() const {
        return std::sqrt(this->squareMagnitude());
    }

    Scalar Vector4::squareMagnitude() const {
        return (this->coordinates.x * this->coordinates.x) + (this->coordinates.y * this->coordinates.y) 
                + (this->coordinates.z * this->coordinates.z) + (this->coordinates.w * this->coordinates.w);
    }

    Vector4 Vector4::normalized() const {
        return *this / this->magnitude();
    }

    void Vector4::normalize() {
        *this /= this->magnitude();
    }

    Scalar Vector4::operator [] (const short& i) const {
        if (i == 0)
            return this->coordinates.x;
        if (i == 1)
            return this->coordinates.y;
        if (i == 2)
            return this->coordinates.z;
        if (i == 3)
            return this->coordinates.w;

        throw std::invalid_argument("INDEX_OUT_OF_RANGE::in Vector4 the index might be either 0, 1, 2 or 3");
    }

    Scalar& Vector4::operator [] (const short& i) {
        if (i == 0)
            return this->coordinates.x;
        if (i == 1)
            return this->coordinates.y;
        if (i == 2)
            return this->coordinates.z;
        if (i == 3)
            return this->coordinates.w;
        
        throw std::invalid_argument("INDEX_OUT_OF_RANGE::in Vector4 the index might be either 0 or 1");
    }

    Scalar Vector4::operator * (const Vector4& vector) const {
        return this->dot(vector);
    }

    Vector4 Vector4::operator + (const Vector4& vector) const {
        return Vector4(this->coordinates.x + vector.coordinates.x, 
            this->coordinates.y + vector.coordinates.y,
            this->coordinates.z + vector.coordinates.z,
            this->coordinates.w + vector.coordinates.w);
    }

    Vector4 Vector4::operator - (const Vector4& vector) const {
            return Vector4(this->coordinates.x - vector.coordinates.x, 
                this->coordinates.y - vector.coordinates.y,
                this->coordinates.z - vector.coordinates.z,
                this->coordinates.w - vector.coordinates.w);
    }

    bool Vector4::operator == (const Vector4& vector) const {
        return this->areEquals(vector);
    }

    Vector4 Vector4::operator - () const {
        return Vector4(-this->coordinates.x, -this->coordinates.y, -this->coordinates.z, -this->coordinates.w);
    }

    Vector4 Vector4::operator * (const Scalar& k) const {
        return Vector4(this->coordinates.x * k, 
                this->coordinates.y * k,
                this->coordinates.z * k,
                this->coordinates.w * k);
    }

    Vector4 Vector4::operator / (const Scalar& k) const {
        return Vector4(this->coordinates.x / k, 
                this->coordinates.y / k,
                this->coordinates.z / k,
                this->coordinates.w / k);
    }

    void Vector4::operator += (const Vector4& vector) {
        this->coordinates.x += vector.coordinates.x;
        this->coordinates.y += vector.coordinates.y;
        this->coordinates.z += vector.coordinates.z;
    }

    void Vector4::operator -= (const Vector4& vector) {
        this->coordinates.x -= vector.coordinates.x;
        this->coordinates.y -= vector.coordinates.y;
        this->coordinates.z -= vector.coordinates.z;
    }

    void Vector4::operator *= (const Scalar& k) {
        this->coordinates.x *= k;
        this->coordinates.y *= k;
        this->coordinates.z *= k;
    }

    void Vector4::operator /= (const Scalar& k) {
        this->coordinates.x /= k;
        this->coordinates.y /= k;
        this->coordinates.z /= k;
    }

    Scalar Vector4::angleBetween (const Vector4& vector) const {
        return Math::radiansToDegreeAngle(std::acos(this->normalized() * vector.normalized()));
    }

    Vector4 Vector4::lerp(const Vector4& vector, const Scalar& t) const {
        return *this * (1 - t) + vector * t;
    }

    Quaternion Vector4::toQuaternion() const {
        return Quaternion(this->coordinates.x, this->coordinates.y, this->coordinates.z, this->coordinates.w);
    }

    bool Vector4::isZero() const {
        return this->squareMagnitude() == 0;
    }

    bool Vector4::areEquals(const Vector4& vector) const{
        return (*this - vector).isZero();
    }

    void Vector4::print() const {
        std::cout << "( " << this->coordinates.x << ", " << this->coordinates.y << ", " << this->coordinates.z << ", " << this->coordinates.w << " )" << std::endl;
    }
}