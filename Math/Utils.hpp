#ifndef UTILS_HPP
#define UTILS_HPP
#define PI 3.14159265358979323846
#include <math.h>
#include "Scalar.hpp"
namespace Athena
{
    class Utils
    {
    public:
        
        static Scalar radians(Scalar degree)
        {
            return (degree * (PI / 180));
        }
    };
}
#endif