#ifndef MATRIX_HPP
#define MATRIX_HPP
#include "../Scalar.hpp"

namespace Athena 
{
    template<int R, int C, class T, typename K>
    class Matrix
    {
    public:
        K data[R * C];
        virtual ~Matrix(){};

        virtual T operator*(const T& mat);
        virtual T operator*(const K& value);
        virtual void operator*=(const T& mat);
        virtual void operator*=(const K& value);

        virtual void operator+=(const T& mat);
        virtual void operator-=(const T& mat);
        virtual T operator+(const T& mat);
        virtual T operator-(const T& mat);

        virtual T operator==(const T& mat);

        virtual void setInverse(const T& mat);
        virtual T inverse() const;
        static T inverse(const T& mat);

        virtual void setTranspose(const T& mat);
        virtual T transpose();
        static T transpose(const T& mat);

        virtual void print();

    };
}

#endif