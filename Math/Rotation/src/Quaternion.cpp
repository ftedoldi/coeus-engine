#include <Quaternion.hpp>

namespace Athena {
    Quaternion::Quaternion() : immaginary(_immaginary), real(_real) {
        this->_immaginary = Vector3(0, 0, 0);
        this->_real = 1.0;
    }

    Quaternion::Quaternion(const Scalar& a, const Scalar& b, const Scalar& c, const Scalar& d) : immaginary(_immaginary), real(_real)  {
        this->_immaginary = Vector3(a, b, c);
        this->_real = d;
    }

    Quaternion::Quaternion(const Vector3& immaginary, const Scalar& real) : immaginary(_immaginary), real(_real)  {
        this->_immaginary = Vector3(immaginary);
        this->_real = real;
    }

    Quaternion::Quaternion(const Vector4& quaternion) : immaginary(_immaginary), real(_real)  {
        this->_immaginary = Vector3(quaternion.coordinates.x, quaternion.coordinates.y, quaternion.coordinates.z);
        this->_real = quaternion.coordinates.w;
    }

    Quaternion::Quaternion(const Quaternion& quaternion) : immaginary(_immaginary), real(_real)  {
        this->_immaginary = quaternion.immaginary;
        this->_real = quaternion.real;
    }

    Quaternion Quaternion::Identity() { return Quaternion(0, 0, 0, 1); }

    Quaternion Quaternion::AxisAngleToQuaternion(const Degree& angle, const Vector3& axis) {
        Scalar angleRad = Math::degreeToRandiansAngle(angle);

        return Quaternion(
            axis * std::sin(angleRad * 0.5f),
            std::cos(angleRad * 0.5f)
        );
    }
    
    void Quaternion::QuaternionToAxisAngle(const Quaternion& quaternion, Scalar& angle, Vector3& axis) {
        Vector3 immm = quaternion.immaginary;
        Scalar real = quaternion.real;

        auto im = immm.coordinates;

        angle = 2.f * std::acos(real);

        axis.coordinates.x = im.x / std::sqrt(1.f - real * real);
        axis.coordinates.y = im.y / std::sqrt(1.f - real * real);
        axis.coordinates.z = im.z / std::sqrt(1.f - real * real);
    }

    Quaternion Quaternion::EulerAnglesToQuaternion(const Vector3& eulerAngles) {
        Vector3 angles = Vector3(eulerAngles);

        angles.coordinates.x = Math::degreeToRandiansAngle(eulerAngles.coordinates.x);
        angles.coordinates.y = Math::degreeToRandiansAngle(eulerAngles.coordinates.y);
        angles.coordinates.z = Math::degreeToRandiansAngle(eulerAngles.coordinates.z);

        Scalar c1 = std::cos(angles.coordinates.y * 0.5);
        Scalar s1 = std::sin(angles.coordinates.y * 0.5);
        Scalar c2 = std::cos(angles.coordinates.z * 0.5);
        Scalar s2 = std::sin(angles.coordinates.z * 0.5);
        Scalar c3 = std::cos(angles.coordinates.x * 0.5);
        Scalar s3 = std::sin(angles.coordinates.x * 0.5);

        Scalar c1c2 = c1 * c2;
        Scalar s1s2 = s1 * s2;

        Quaternion q = Quaternion(
            c1c2 * s3 + s1s2 * c3,
            s1 * c2 * c3 + c1 * s2 * s3,
            c1 * s2 * c3 - s1 * c2 * s3,
            c1c2 * c3 - s1s2 * s3
        );

        return q.normalized();
    }

    Vector3 Quaternion::QuaternionToEulerAngles(const Quaternion& quaternion) {
        Vector3 result = Vector3();
        Vector4 quat = Vector4(quaternion.immaginary.coordinates.x, quaternion.immaginary.coordinates.y, quaternion.immaginary.coordinates.z, quaternion.real);

        auto q = quat.coordinates;
        
        Scalar singularityAtNorthPole = 0.499;
        Scalar singularityAtSouthPole = -0.499;

        Scalar test = q.x * q.y + q.z * q.w;

        if (test > singularityAtNorthPole)
            return Vector3(Math::radiansToDegreeAngle(0), Math::radiansToDegreeAngle(2 * std::atan2(q.x, q.w)), Math::radiansToDegreeAngle(M_PI / 2));
        
        if (test < singularityAtSouthPole)
            return Vector3(Math::radiansToDegreeAngle(0), Math::radiansToDegreeAngle(-2 * std::atan2(q.x, q.w)), Math::radiansToDegreeAngle(-M_PI / 2));

        double squaredX = q.x * q.x;
        double squaredY = q.y * q.y;
        double squaredZ = q.z * q.z;

        result.coordinates.y = Math::radiansToDegreeAngle(std::atan2(2.f * q.y * q.w - 2.f * q.x * q.z, 1 - 2.f * (squaredY * squaredZ)));
        result.coordinates.z = Math::radiansToDegreeAngle(std::asin(2.f * test));
        result.coordinates.x = Math::radiansToDegreeAngle(std::atan2(2.f * q.x * q.w - 2.f * q.y * q.z, 1 - 2.f * (squaredX * squaredZ)));

        return result;
    }

    Quaternion Quaternion::Matrix3ToQuaternion(const Matrix3& matrix) {
        auto vecMatrix = matrix.asVector3Array();

        auto r0 = vecMatrix.row0;
        auto r1 = vecMatrix.row1;
        auto r2 = vecMatrix.row2;

        float realPart = std::sqrt(1 + r0[0] + r1[1] + r2[2]) / 2.f;

        return Quaternion(
            (r2[1] - r1[2]) / (4.f * realPart),
            (r0[2] - r2[0]) / (4.f * realPart),
            (r1[0] - r0[1]) / (4.f * realPart),
            realPart
        );
    }

    Quaternion Quaternion::Matrix4ToQuaternion(const Matrix4& matrix)
    {
        Quaternion q;
        q.real = std::sqrt(std::max(0.0f, 1 + matrix.data[0] + matrix.data[5] + matrix.data[10])) / 2;
        q.immaginary.coordinates.x = std::sqrt(std::max(0.0f, 1 + matrix.data[0] - matrix.data[5] - matrix.data[10])) / 2;
        q.immaginary.coordinates.y = std::sqrt(std::max(0.0f, 1 - matrix.data[0] + matrix.data[5] - matrix.data[10])) / 2;
        q.immaginary.coordinates.z = std::sqrt(std::max(0.0f, 1 - matrix.data[0] - matrix.data[5] + matrix.data[10])) / 2;
        int signX = (q.immaginary.coordinates.x * (matrix.data[9] - matrix.data[6])) > 0 ? 1 : ((q.immaginary.coordinates.x * (matrix.data[9] - matrix.data[6]) < 0) ? -1 : 0);
        int signY = (q.immaginary.coordinates.y * (matrix.data[2] - matrix.data[8])) > 0 ? 1 : ((q.immaginary.coordinates.x * (matrix.data[2] - matrix.data[8]) < 0) ? -1 : 0);
        int signZ = (q.immaginary.coordinates.z * (matrix.data[4] - matrix.data[1])) > 0 ? 1 : ((q.immaginary.coordinates.x * (matrix.data[4] - matrix.data[1]) < 0) ? -1 : 0);
        q.immaginary.coordinates.x *= signX;
        q.immaginary.coordinates.y *= signY;
        q.immaginary.coordinates.z *= signZ;
        return q;
    }

    Quaternion Quaternion::matToQuatCast(Matrix4& matrix)
    {
        Matrix3 mat = Matrix4::toMatrix3(matrix);

        Scalar fourXSquaredMinus1 = mat.data[0] - mat.data[4] - mat.data[8];
		Scalar fourYSquaredMinus1 = mat.data[4] - mat.data[0] - mat.data[8];
		Scalar fourZSquaredMinus1 = mat.data[8] - mat.data[0] - mat.data[4];
		Scalar fourWSquaredMinus1 = mat.data[0] + mat.data[4] + mat.data[8];

        int biggestIndex = 0;
		Scalar fourBiggestSquaredMinus1 = fourWSquaredMinus1;
		if(fourXSquaredMinus1 > fourBiggestSquaredMinus1)
		{
			fourBiggestSquaredMinus1 = fourXSquaredMinus1;
			biggestIndex = 1;
		}
		if(fourYSquaredMinus1 > fourBiggestSquaredMinus1)
		{
			fourBiggestSquaredMinus1 = fourYSquaredMinus1;
			biggestIndex = 2;
		}
		if(fourZSquaredMinus1 > fourBiggestSquaredMinus1)
		{
			fourBiggestSquaredMinus1 = fourZSquaredMinus1;
			biggestIndex = 3;
		}

        Scalar biggestVal = sqrt(fourBiggestSquaredMinus1 + static_cast<Scalar>(1)) * static_cast<Scalar>(0.5);
		Scalar mult = static_cast<Scalar>(0.25) / biggestVal;

        switch(biggestIndex)
		{
		case 0:
            return Quaternion((mat.data[5] - mat.data[7]) * mult, (mat.data[6] - mat.data[2]) * mult, (mat.data[1] - mat.data[3]) * mult, biggestVal);
		case 1:
            return Quaternion(biggestVal, (mat.data[1] + mat.data[3]) * mult, (mat.data[6] + mat.data[2]) * mult, (mat.data[5] - mat.data[7]) * mult);
		case 2:
            return Quaternion((mat.data[1] + mat.data[3]) * mult, biggestVal, (mat.data[5] + mat.data[7]) * mult, (mat.data[6] - mat.data[2]) * mult);
		case 3:
            return Quaternion((mat.data[6] + mat.data[2]) * mult, (mat.data[5] + mat.data[7]) * mult, biggestVal, (mat.data[1] - mat.data[3]) * mult);
		default:
			assert(false);
			return Quaternion(0, 0, 0, 1);
		}
    }

    Matrix3 Quaternion::QuaternionToMatrx3(const Quaternion& quaternion) {
        Vector4 quat = Quaternion::AsVector4(quaternion.normalized());

        auto q = quat.coordinates;

        Scalar sqw = q.w * q.w;
        Scalar sqx = q.x * q.x;
        Scalar sqy = q.y * q.y;
        Scalar sqz = q.z * q.z;

        Scalar tmp1 = q.x * q.y;
        Scalar tmp2 = q.z * q.w;
        Scalar tmp11 = q.x * q.z;
        Scalar tmp21 = q.y * q.w;
        Scalar tmp12 = q.y * q.z;
        Scalar tmp22 = q.x * q.w;

        Scalar inverse = 1 / (sqx + sqy + sqz + sqw);

        return Matrix3 (
            (sqx - sqy - sqz + sqw) * inverse, 2.f * (tmp1 - tmp2) * inverse, 2.f * (tmp11 + tmp21) * inverse,
            2.f * (tmp1 + tmp2) * inverse, (-sqx + sqy - sqz + sqw) * inverse, 2.f * (tmp12 - tmp22) * inverse,
            2.f * (tmp11 - tmp21) * inverse, 2.f * (tmp12 + tmp22) * inverse, (-sqx - sqy + sqz + sqw) * inverse
        );
    }

    Vector4 Quaternion::AsVector4(const Quaternion& quaternion) {
        return Vector4(quaternion.immaginary, quaternion.real);
    }

    Quaternion Quaternion::RotationBetweenVectors(const Vector3& start, const Vector3& destination) {
        Vector3 s = start.normalized();
        Vector3 d = destination.normalized();

        Scalar cosTheta = s.dot(d);
        Vector3 rotationAxis;

        if (cosTheta < -1 + 0.001f) {
            rotationAxis = Vector3::cross(Vector3::forward(), s);

            if (rotationAxis.squareMagnitude() < 0.01)
                rotationAxis = Vector3::cross(Vector3::right(), s);

            rotationAxis.normalize();
            return AxisAngleToQuaternion(Math::degreeToRandiansAngle(180.0f), rotationAxis);
        }

        rotationAxis = Vector3::cross(s, d);

        float sqrt = std::sqrt( (1 + cosTheta) * 2);
        float inverseSqrt = 1 / sqrt;

        return Quaternion(rotationAxis.coordinates.x * inverseSqrt, rotationAxis.coordinates.y * inverseSqrt, rotationAxis.coordinates.z * inverseSqrt, sqrt * 0.5f);
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
        this->_immaginary = -this->_immaginary;
    }

    Quaternion Quaternion::conjugated() const {
        return Quaternion(-this->_immaginary, this->_real);
    }

    Scalar Quaternion::squareMagnitude() const {
        return this->immaginary.squareMagnitude() + this->_real * this->_real;
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

    void Quaternion::operator = (const Quaternion& quaternion)
    {
        this->_immaginary = quaternion.immaginary;
        this->_real = quaternion.real;
    }

    void Quaternion::operator /=(const Scalar& k) {
        this->_immaginary /= k;
        this->_real /= k;
    }

    void Quaternion::operator *=(const Scalar& k) {
        this->_immaginary *= k;
        this->_real *= k;
    }

    void Quaternion::operator +=(const Quaternion& quaternion) {
        this->_immaginary += quaternion._immaginary;
        this->_real += quaternion._real;
    }

    void Quaternion::operator -=(const Quaternion& quaternion) {
        this->_immaginary -= quaternion._immaginary;
        this->_real -= quaternion._real;
    }

    bool Quaternion::operator ==(const Quaternion& quaternion) const {
        return (this->_immaginary == quaternion.immaginary && this->_real == quaternion.real);
    }

    bool Quaternion::operator !=(const Quaternion& quaternion) const {
        return !((*this) == quaternion);
    }

    Vector3 Quaternion::rotateVectorByThisQuaternion(const Vector3& vectorToRotate) const {
        Quaternion* result = new Quaternion(this->conjugated() * Quaternion(vectorToRotate, 0.0) * (*this));

        return result->immaginary;
    }

    Quaternion Quaternion::normalized() const {
        return Quaternion(this->immaginary / this->magnitude(), this->real / this->magnitude());
    }

    Vector4 Quaternion::asVector4() const {
        return Quaternion::AsVector4(*this);
    }

    void Quaternion::print() const {
        std::cout << "( " << this->_immaginary.coordinates.x << ", " << this->_immaginary.coordinates.y << ", " 
            << this->_immaginary.coordinates.z << ", " << this->_real << " )" << std::endl;
    }

    Quaternion operator *(const Quaternion& a, const Quaternion& b) {
        return Quaternion(
            a.real * b.immaginary + a.immaginary * b.real + Vector3::cross(a.immaginary, b.immaginary),  // imaginary part
            a.real * b.real - Vector3::dot(a.immaginary, b.immaginary) // _real part
        );
    }
}