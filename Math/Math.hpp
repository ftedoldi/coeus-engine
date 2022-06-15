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
                return angle *180 / (M_PI);
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
    };
}

#endif