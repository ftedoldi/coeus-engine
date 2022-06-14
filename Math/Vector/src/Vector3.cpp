#include <Vector3.hpp>

namespace Athena {
    Vector3::Vector3() {
        coordinates = Vector3Coordinates<Scalar>();
        this->coordinates.x = (Scalar) 0.0;
        this->coordinates.y = (Scalar) 0.0;
        this->coordinates.z = (Scalar) 0.0;
    }

    Vector3::Vector3(const Scalar& x, const Scalar& y, const Scalar& z) {
        coordinates = Vector3Coordinates<Scalar>();
        this->coordinates.x = x;
        this->coordinates.y = y;
        this->coordinates.z = z;
    }

    Vector3::Vector3(const Vector2& vector, const Scalar& z) {
        coordinates = Vector3Coordinates<Scalar>();
        this->coordinates.x = vector.coordinates.x;
        this->coordinates.y = vector.coordinates.y;
        this->coordinates.z = z;
    }

    Vector3::Vector3(const Vector3& vector) {
        coordinates = Vector3Coordinates<Scalar>();
        this->coordinates.x = vector.coordinates.x;
        this->coordinates.y = vector.coordinates.y;
        this->coordinates.z = vector.coordinates.z;
    }

    Scalar Vector3::dot(const Vector3& vector) const {
        return this->coordinates.x * vector.coordinates.x +
                this->coordinates.y * vector.coordinates.y +
                this->coordinates.z * vector.coordinates.z;
    }

    Vector3 Vector3::cross(const Vector3& vector) const {
        return Vector3::cross(*this, vector);
    }

    Scalar Vector3::magnitude() const {
        return std::sqrt(this->squareMagnitude());
    }

    Scalar Vector3::squareMagnitude() const {
        return (this->coordinates.x * this->coordinates.x) + (this->coordinates.y * this->coordinates.y) + (this->coordinates.z * this->coordinates.z);
    }

    Vector3 Vector3::normalized() const {
        return *this / this->magnitude();
    }

    void Vector3::normalize() {
        *this /= this->magnitude();
    }

    Scalar Vector3::operator [] (const short& i) const {
        if (i == 0)
            return this->coordinates.x;
        if (i == 1)
            return this->coordinates.y;
        if (i == 2)
            return this->coordinates.z;

        throw std::invalid_argument("INDEX_OUT_OF_RANGE::in Vector3 the index might be either 0 or 1");
    }

    Scalar& Vector3::operator [] (const short& i) {
        if (i == 0)
            return this->coordinates.x;
        if (i == 1)
            return this->coordinates.y;
        if (i == 2)
            return this->coordinates.z;
        
        throw std::invalid_argument("INDEX_OUT_OF_RANGE::in Vector3 the index might be either 0 or 1");
    }

    Scalar Vector3::operator * (const Vector3& vector) const {
        return this->dot(vector);
    }

    Vector3 Vector3::operator + (const Vector3& vector) const {
        return Vector3(this->coordinates.x + vector.coordinates.x, 
            this->coordinates.y + vector.coordinates.y,
            this->coordinates.z + vector.coordinates.z);
    }

    Vector3 Vector3::operator - (const Vector3& vector) const {
            return Vector3(this->coordinates.x - vector.coordinates.x, 
                this->coordinates.y - vector.coordinates.y,
                this->coordinates.z - vector.coordinates.z);
    }

    bool Vector3::operator == (const Vector3& vector) const {
        return this->areEquals(vector);
    }

    Vector3 Vector3::operator - () const {
        return Vector3(-this->coordinates.x, -this->coordinates.y, -this->coordinates.z);
    }

    Vector3 Vector3::operator * (const Scalar& k) const {
        return Vector3(this->coordinates.x * k, 
                this->coordinates.y * k,
                this->coordinates.z * k);
    }

    Vector3 Vector3::operator / (const Scalar& k) const {
        return Vector3(this->coordinates.x / k, 
                this->coordinates.y / k,
                this->coordinates.z / k);
    }

    void Vector3::operator += (const Vector3& vector) {
        this->coordinates.x += vector.coordinates.x;
        this->coordinates.y += vector.coordinates.y;
        this->coordinates.z += vector.coordinates.z;
    }

    void Vector3::operator -= (const Vector3& vector) {
        this->coordinates.x -= vector.coordinates.x;
        this->coordinates.y -= vector.coordinates.y;
        this->coordinates.z -= vector.coordinates.z;
    }

    void Vector3::operator *= (const Scalar& k) {
        this->coordinates.x *= k;
        this->coordinates.y *= k;
        this->coordinates.z *= k;
    }

    void Vector3::operator /= (const Scalar& k) {
        this->coordinates.x /= k;
        this->coordinates.y /= k;
        this->coordinates.z /= k;
    }

    // TODO: Convert cosine from radians to degree
    Scalar Vector3::angleBetween (const Vector3& vector) const {
        return std::acos(this->normalized() * vector.normalized());
    }

    Vector3 Vector3::lerp(const Vector3& vector, const Scalar& t) const {
        return *this * (1 - t) + vector * t;
    }

    bool Vector3::isZero() const {
        return this->squareMagnitude() == 0;
    }

    bool Vector3::areEquals(const Vector3& vector) const{
        return (*this - vector).isZero();
    }

    void Vector3::print() const {
        std::cout << "( " << this->coordinates.x << ", " << this->coordinates.y << ", " << this->coordinates.z << " )" << std::endl;
    }

    Vector3 operator * (const Scalar& k, const Vector3& vector) {
        return vector * k;
    }
}