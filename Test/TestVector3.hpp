#pragma once

#include <Test.hpp>
#include <Vector2.hpp>
#include <Vector3.hpp>

#include <cmath>
#include <stdexcept>

using namespace Athena;

class TestVector3 : public Test {
    private:
        void testConstructor() {
            Vector3 a = Vector3(1, 2, 10);
            assert(a.coordinates.x == 1);
            assert(a.coordinates.y == 2);
            assert(a.coordinates.z == 10);

            Vector3 b = Vector3();
            assert(b.coordinates.x == 0);
            assert(b.coordinates.y == 0);
            assert(b.coordinates.z == 0);

            Vector3 c = Vector3(Vector3(2, 4, 10));
            assert(c.coordinates.x == 2);
            assert(c.coordinates.y == 4);
            assert(c.coordinates.z == 10);

            Vector3 d = Vector3(Vector2(2, 4), 10);
            assert(d.coordinates.x == 2);
            assert(d.coordinates.y == 4);
            assert(d.coordinates.z == 10);

            assert(Vector3() == Vector3(0, 0, 0));
            assert(Vector3(Vector3(2, 3, 10)) == Vector3(2, 3, 10));
            assert(Vector3(Vector3(2, 3, 10)) == Vector3(Vector2(2, 3), 10));
        }

        void testDotProduct() {
            Vector3 a = Vector3(2, 2, 2);
            assert(a.dot(a) == a.squareMagnitude());
            assert(a.dot(Vector3(0, 0, 0)) == 0.f);
            assert(a.dot(Vector3(1, 3, 0)) == 8.f); 
        }

        void testCrossProduct() {
            Vector3 a = Vector3(1, 1, 0);
            assert(a.cross(Vector3(1, 2, 0)) == Vector3(0, 0, 1));
        }

        void testMagnitude() {
            Vector3 a = Vector3(2, 3, 0);
            assert(a.squareMagnitude() == 13);
            assert(a.magnitude() == std::sqrt(a.squareMagnitude()));
            assert(Vector3().magnitude() == 0);

            Vector3 b = Vector3(1, 0, 0);
            assert(b.squareMagnitude() == Vector3(0, 1, 0).squareMagnitude());
        }

        void testNormalize() {
            Vector3 a = Vector3(1, 0, 0);
            assert(a.normalized().squareMagnitude() == 1);
            assert(a.normalized() == a);
            a.normalize();
            assert(a.normalized() == a);

            Vector3 b = Vector3(2, 0, 0);
            assert(b.normalized() == a);
            b.normalize();
            assert(b == a);  
        }

        void testOperators() {
            Vector3 a = Vector3(10.432f, 3200.26890f, 15.2535f);
            assert(a[0] == 10.432f);
            assert(a[1] == 3200.26890f);
            assert(a[2] == 15.2535f);
            bool exception = false;
            try {
                auto tmp = a[3];
            } catch (...) {
                exception = true;
            }
            assert(exception == true);
            a[0] = 1;
            a[1] = 2;
            a[2] = 0;
            assert(a[0] == 1);
            assert(a[1] == 2);
            assert(a[2] == 0);
            assert((a * Vector3(2, 4, 0)) == a.dot(Vector3(2, 4, 0)));
            assert((a + Vector3(1, 1, 0)) == Vector3(2, 3, 0));
            assert((a - Vector3(1, 1, 0)) == Vector3(0, 1, 0));
            assert(-a == Vector3(-1, -2, 0));
            assert((a * 2) == Vector3(2, 4, 0));
            assert((a / 2) == Vector3(0.5f, 1, 0));

            a += Vector3(1, 1, 0);
            assert(a == Vector3(2, 3, 0));
            a -= Vector3(1, 1, 0);
            assert(a == Vector3(1, 2, 0));
            a *= 2.f;
            assert(a == Vector3(2, 4, 0));
            a /= 2.f;
            assert(a == Vector3(1, 2, 0));
        }

        void testAngleBetween() {
            Vector3 a = Vector3(1, 0, 0);
            Vector3 b = Vector3(0, 1, 0);
            assert(a.angleBetween(b) == 90.f);
            assert(b.angleBetween(Vector3(0.5f, 0.5f, 0)) == 45.f);
        }

        void testLerp() {
            Vector3 a = Vector3(1, 1, 0);
            assert(a.lerp(Vector3(1, 0, 0), 0.5f) == Vector3(1, 0.5f, 0));
            assert(a.lerp(Vector3(0, 0, 0), 0.5f) == Vector3(0.5f, 0.5f, 0));
            assert(a.lerp(Vector3(0, 0, 0), 0.f) == a);
            assert(a.lerp(Vector3(0, 0, 0), 1.f) == Vector3(0, 0, 0));
        }

        void testEquals() {
            Vector3 a = Vector3();
            assert(a.isZero());
            assert(!a.areEquals(Vector3(0, 0.1f, 0)));
            assert(a.areEquals(Vector3()));
        }

    public:
        TestVector3() {}

        void test() {
            testConstructor();
            testDotProduct();
            testCrossProduct();
            testMagnitude();
            testNormalize();
            testOperators();
            testAngleBetween();
            testLerp();
            testEquals();
        }
};