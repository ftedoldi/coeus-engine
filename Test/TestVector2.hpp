#pragma once

#include <Test.hpp>
#include <Vector2.hpp>
#include <Vector3.hpp>

#include <cmath>
#include <stdexcept>

using namespace Athena;

class TestVector2 : public Test {
    private:
        void testConstructor() {
            Vector2 a = Vector2(1, 2);
            assert(a.coordinates.x == 1);
            assert(a.coordinates.y == 2);

            Vector2 b = Vector2();
            assert(b.coordinates.x == 0);
            assert(b.coordinates.y == 0);

            Vector2 c = Vector2(Vector2(2, 4));
            assert(c.coordinates.x == 2);
            assert(c.coordinates.y == 4);

            assert(Vector2() == Vector2(0, 0));
            assert(Vector2(Vector2(2, 3)) == Vector2(2, 3));
        }

        void testDotProduct() {
            Vector2 a = Vector2(2, 2);
            assert(a.dot(a) == a.squareMagnitude());
            assert(a.dot(Vector2(0, 0)) == 0.f);
            assert(a.dot(Vector2(1, 3)) == 8.f); 
        }

        void testCrossProduct() {
            Vector2 a = Vector2(1, 1);
            assert(a.cross(Vector2(1, 2)) == Vector3(0, 0, 1));
        }

        void testMagnitude() {
            Vector2 a = Vector2(2, 3);
            assert(a.squareMagnitude() == 13);
            assert(a.magnitude() == std::sqrt(a.squareMagnitude()));
            assert(Vector2().magnitude() == 0);

            Vector2 b = Vector2(1, 0);
            assert(b.squareMagnitude() == Vector2(0, 1).squareMagnitude());
        }

        void testNormalize() {
            Vector2 a = Vector2(1, 0);
            assert(a.normalized().squareMagnitude() == 1);
            assert(a.normalized() == a);
            a.normalize();
            assert(a.normalized() == a);

            Vector2 b = Vector2(2, 0);
            assert(b.normalized() == a);
            b.normalize();
            assert(b == a);  
        }

        void testOperators() {
            Vector2 a = Vector2(10.432f, 3200.26890f);
            assert(a[0] == 10.432f);
            assert(a[1] == 3200.26890f);
            bool exception = false;
            try {
                auto tmp = a[2];
            } catch (...) {
                exception = true;
            }
            assert(exception == true);
            a[0] = 1;
            a[1] = 2;
            assert(a[0] == 1);
            assert(a[1] == 2);
            assert((a * Vector2(2, 4)) == a.dot(Vector2(2, 4)));
            assert((a + Vector2(1, 1)) == Vector2(2, 3));
            assert((a - Vector2(1, 1)) == Vector2(0, 1));
            assert(-a == Vector2(-1, -2));
            assert((a * 2) == Vector2(2, 4));
            assert((a / 2) == Vector2(0.5f, 1));

            a += Vector2(1, 1);
            assert(a == Vector2(2, 3));
            a -= Vector2(1, 1);
            assert(a == Vector2(1, 2));
            a *= 2.f;
            assert(a == Vector2(2, 4));
            a /= 2.f;
            assert(a == Vector2(1, 2));
        }

        void testAngleBetween() {
            Vector2 a = Vector2(1, 0);
            Vector2 b = Vector2(0, 1);
            assert(a.angleBetween(b) == 90.f);
            assert(b.angleBetween(Vector2(0.5f, 0.5f)) == 45.f);
        }

        void testLerp() {
            Vector2 a = Vector2(1, 1);
            assert(a.lerp(Vector2(1, 0), 0.5f) == Vector2(1, 0.5f));
            assert(a.lerp(Vector2(0, 0), 0.5f) == Vector2(0.5f, 0.5f));
            assert(a.lerp(Vector2(0, 0), 0.f) == a);
            assert(a.lerp(Vector2(0, 0), 1.f) == Vector2(0, 0));
        }

        void testEquals() {
            Vector2 a = Vector2();
            assert(a.isZero());
            assert(!a.areEquals(Vector2(0, 0.1f)));
            assert(a.areEquals(Vector2()));
        }

    public:
        TestVector2() {}

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