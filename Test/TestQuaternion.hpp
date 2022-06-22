#pragma once

#include <Test.hpp>
#include <Vector3.hpp>
#include <Vector4.hpp>
#include <Matrix3.hpp>
#include <Quaternion.hpp>

#include <cmath>
#include <stdexcept>

using namespace Athena;

#define EPSILON 0.0001

class TestQuaternion : public Test {
    private:
        void testConstructor() {
            Quaternion();
            Quaternion a = Quaternion(1, 2, 3, 4);
            Quaternion b = Quaternion(Vector3(1, 2, 3), 4);
            Quaternion c = Quaternion(Vector4(1, 2, 3, 4));
            Quaternion d = Quaternion(Quaternion(1, 2, 3, 4));

            assert(Quaternion().asVector4() == Vector4(0, 0, 0, 1));
            assert(a == b);
            assert(b == c);
            assert(c == d);
            assert(d == a);
            assert(Quaternion() == Quaternion::Identity());
        }

        void testAxisAngle() {
            Quaternion q = Quaternion::AxisAngleToQuaternion(171.887, Vector3(1, 0, 0));
            assert(q.asVector4().coordinates.x - 0.997495f < EPSILON);
            assert(q.asVector4().coordinates.y < EPSILON);
            assert(q.asVector4().coordinates.z < EPSILON);
            assert(q.asVector4().coordinates.w - 0.0707372 < EPSILON);

            Vector3 axis = Vector3();
            Scalar angle;
            Quaternion::QuaternionToAxisAngle(q, angle, axis);
            assert(axis.coordinates.x - EPSILON <= 1);
            assert(axis.coordinates.y <= 0);
            assert(axis.coordinates.z <= 0);
            assert(angle <= 172.0f);
        }

        void testEulerAngles() {
            Quaternion q = Quaternion::EulerAnglesToQuaternion(Vector3(10, 20, 40));
            q.asVector4().print();
            Quaternion::QuaternionToEulerAngles(q).print();
        }

        void testMatrix3() {
            Quaternion q = Quaternion::Matrix3ToQuaternion(Matrix3(0, -1, 0, 1, 0, 0, 0, 0, 1));
            q.asVector4().print();
            Quaternion::QuaternionToMatrx3(q).print();
        }

        void testConjugate() {
            assert(Quaternion(1, 1, 2, 3).conjugated().asVector4() == Vector4(-1, -1, -2, 3));
            Quaternion q = Quaternion(14.f, 2.45f, 6.7f, 40.23534f);
            q.conjugate();
            assert(q.asVector4() == Vector4(-14.f, -2.45f, -6.7f, 40.23534f));
        }

        void testMagnitude() {
            Quaternion q = Quaternion(1, 1, 1, 2);
            assert(q.squareMagnitude() == Vector4(1, 1, 1, 2).squareMagnitude());
            assert(q.squareMagnitude() == 7);
            assert(q.magnitude() == Vector4(1, 1, 1, 2).magnitude());
            assert(q.magnitude() == std::sqrt(Vector4(1, 1, 1, 2).squareMagnitude()));
        }

    public:
        TestQuaternion() {}

        void test() {
            testConstructor();
            testAxisAngle();
            testEulerAngles();
            testMatrix3();
            testConjugate();
            testMagnitude();
        }

        /*
        void invert();
        Quaternion inverse() const;

        Quaternion operator /(const Scalar& k) const;
        Quaternion operator *(const Scalar& k) const;

        void operator /=(const Scalar& k);
        void operator *=(const Scalar& k);
        void operator +=(const Quaternion& quaternion);
        void operator -=(const Quaternion& quaternion);

        bool operator ==(const Quaternion& quaternion) const;
        bool operator !=(const Quaternion& quaternion) const;

        // Rotate the vector passed as input by this quaternion
        Vector3 rotateVectorByThisQuaternion(const Vector3& vectorToRotate);

        Vector4 asVector4() const;

        void print() const;
        */
};