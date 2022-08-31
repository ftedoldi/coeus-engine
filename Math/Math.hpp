#ifndef MATH_HPP
#define MATH_HPP

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

#include <cmath>
#include <iostream>

namespace Athena
{
    typedef float Scalar;

    const Scalar EPSILON = 1e-10;

    class Math
    {
        public:
            static Scalar radiansToDegreeAngle(const Scalar& angle) {
                return angle * 180 / (M_PI);
            }

            static Scalar degreeToRandiansAngle(const Scalar& angle) {
                return angle * (M_PI) / 180;
            }
            
            static Scalar getAngleInRadiansFromCos(const Scalar& cosine) {
                return std::acos(cosine);
            }

            static Scalar getAngleInRadiansFromSin(const Scalar& sine) {
                return std::asin(sine);
            }

            static Scalar areEquals(const Scalar& a, const Scalar& b) {
                return std::abs(a - b) < EPSILON;
            }

            static Scalar clamp(const Scalar& start, const Scalar& end, const Scalar& value) {
                return value < start ? start : value > end ? end : value;
            }

            static Scalar clamp(const int& start, const int& end, const int& value) {
                return value < start ? start : value > end ? end : value;
            }

            static Scalar clamp01(const Scalar& value) {
                return value < 0 ? 0 : value > 1 ? 1 : value;
            }

            static Scalar clamp01(const int& value) {
                return value < 0 ? 0 : value > 1 ? 1 : value;
            }

            static Scalar inverseSqrt(const Scalar& value)
            {
                return static_cast<Scalar>(1) / static_cast<Scalar>(std::sqrt(static_cast<double>(value)));
            }

            // Linear interpolation: where t = 0 = a and t = 1 = b
            static Scalar lerp(const Scalar& a, const Scalar& b, const Scalar& t)
            {
                return (1.0f - t) * a + t * b;
            }

            // Linear interpolation: where t = a = 0 and t = b = 1
            static Scalar inverseLerp(const Scalar& a, const Scalar& b, const Scalar& t)
            {
                return (t - a) / (b - a);
            }

            static Scalar scalarAbs(const Scalar& sc)
            {
                return fabs(sc);
            }

            static Scalar scalarPow(const Scalar& sc, const Scalar& dt)
            {
                return powf(sc, dt);
            }

            static Scalar scalarSqrt(const Scalar& sc)
            {
                return sqrtf(sc);
            }
    };
}

#endif