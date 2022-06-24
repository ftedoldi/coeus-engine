#pragma once

#include <Test.hpp>
#include <Vector2.hpp>
#include <Vector3.hpp>
#include <Vector4.hpp>

#include <cmath>
#include <stdexcept>

using namespace Athena;

class TestVector4 : public Test {
    private:
        void testConstructor() {
            Vector4 a = Vector4(1, 2, 10, 34);
            assert(a.coordinates.x == 1);
            assert(a.coordinates.y == 2);
            assert(a.coordinates.z == 10);
            assert(a.coordinates.w == 34);

            Vector4 b = Vector4();
            assert(b.coordinates.x == 0);
            assert(b.coordinates.y == 0);
            assert(b.coordinates.z == 0);
            assert(b.coordinates.w == 0);

            Vector4 c = Vector4(Vector4(2, 4, 10, 40));
            assert(c.coordinates.x == 2);
            assert(c.coordinates.y == 4);
            assert(c.coordinates.z == 10);
            assert(c.coordinates.w == 40);

            Vector4 d = Vector4(Vector2(2, 4), 10, 50);
            assert(d.coordinates.x == 2);
            assert(d.coordinates.y == 4);
            assert(d.coordinates.z == 10);
            assert(d.coordinates.w == 50);

            Vector4 e = Vector4(Vector3(2, 4, 20), 50);
            assert(e.coordinates.x == 2);
            assert(e.coordinates.y == 4);
            assert(e.coordinates.z == 20);
            assert(e.coordinates.w == 50);

            assert(Vector4() == Vector4(0, 0, 0, 0));
            assert(Vector4(Vector3(2, 3, 10), 40) == Vector4(2, 3, 10, 40));
            assert(Vector4(Vector3(2, 3, 10), 32) == Vector4(Vector2(2, 3), 10, 32));
        }

        void testDotProduct() {
            Vector4 a = Vector4(2, 2, 2, 2);
            assert(a.dot(a) == a.squareMagnitude());
            assert(a.dot(Vector4(0, 0, 0, 0)) == 0.f);
            assert(a.dot(Vector4(1, 3, 0, 0)) == 8.f); 
        }

        void testMagnitude() {
            Vector4 a = Vector4(2, 3, 0, 0);
            assert(a.squareMagnitude() == 13);
            assert(a.magnitude() == std::sqrt(a.squareMagnitude()));
            assert(Vector4().magnitude() == 0);

            Vector4 b = Vector4(1, 0, 0, 0);
            assert(b.squareMagnitude() == Vector4(0, 1, 0, 0).squareMagnitude());
        }

        void testNormalize() {
            Vector4 a = Vector4(1, 0, 0, 0);
            assert(a.normalized().squareMagnitude() == 1);
            assert(a.normalized() == a);
            a.normalize();
            assert(a.normalized() == a);

            Vector4 b = Vector4(2, 0, 0, 0);
            assert(b.normalized() == a);
            b.normalize();
            assert(b == a);  
        }

        void testOperators() {
            Vector4 a = Vector4(10.432f, 3200.26890f, 15.2535f, 34.23415f);
            assert(a[0] == 10.432f);
            assert(a[1] == 3200.26890f);
            assert(a[2] == 15.2535f);
            assert(a[3] == 34.23415f);
            bool exception = false;
            try {
                auto tmp = a[4];
            } catch (...) {
                exception = true;
            }
            assert(exception == true);
            a[0] = 1;
            a[1] = 2;
            a[2] = 0;
            a[3] = 0;
            assert(a[0] == 1);
            assert(a[1] == 2);
            assert(a[2] == 0);
            assert(a[3] == 0);
            assert((a * Vector4(2, 4, 0, 0)) == a.dot(Vector4(2, 4, 0, 0)));
            assert((a + Vector4(1, 1, 0, 0)) == Vector4(2, 3, 0, 0));
            assert((a - Vector4(1, 1, 0, 0)) == Vector4(0, 1, 0, 0));
            assert(-a == Vector4(-1, -2, 0, 0));
            assert((a * 2) == Vector4(2, 4, 0, 0));
            assert((a / 2) == Vector4(0.5f, 1, 0, 0));

            a += Vector4(1, 1, 0, 0);
            assert(a == Vector4(2, 3, 0, 0));
            a -= Vector4(1, 1, 0, 0);
            assert(a == Vector4(1, 2, 0, 0));
            a *= 2.f;
            assert(a == Vector4(2, 4, 0, 0));
            a /= 2.f;
            assert(a == Vector4(1, 2, 0, 0));
        }

        void testAngleBetween() {
            Vector4 a = Vector4(1, 0, 0, 0);
            Vector4 b = Vector4(0, 1, 0, 0);
            assert(a.angleBetween(b) == 90.f);
            assert(b.angleBetween(Vector4(0.5f, 0.5f, 0, 0)) == 45.f);
        }

        void testLerp() {
            Vector4 a = Vector4(1, 1, 0, 0);
            assert(a.lerp(Vector4(1, 0, 0, 0), 0.5f) == Vector4(1, 0.5f, 0, 0));
            assert(a.lerp(Vector4(0, 0, 0, 0), 0.5f) == Vector4(0.5f, 0.5f, 0, 0));
            assert(a.lerp(Vector4(0, 0, 0, 0), 0.f) == a);
            assert(a.lerp(Vector4(0, 0, 0, 0), 1.f) == Vector4(0, 0, 0, 0));
        }

        void testEquals() {
            Vector4 a = Vector4();
            assert(a.isZero());
            assert(!a.areEquals(Vector4(0, 0.1f, 0, 0)));
            assert(a.areEquals(Vector4()));
        }

        void testToQuaternion() {
            Vector4 a = Vector4(0, 0, 0, 1);
            assert(a.toQuaternion().immaginary == Vector3());
            assert(a.toQuaternion().real == 1);
        }

    public:
        TestVector4() {}

        void test() {
            testConstructor();
            testDotProduct();
            testMagnitude();
            testNormalize();
            testOperators();
            testAngleBetween();
            testLerp();
            testToQuaternion();
            testEquals();
        }
};