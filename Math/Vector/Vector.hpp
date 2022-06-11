#ifndef VECTOR_HPP
#define VECTOR_HPP

namespace Athena
{
    template<typename S, class T, typename K>
    class Vector
    {
        public:
            S coordinates;

            virtual ~Vector() {}

            static K dot(const T& vector1, const T& vector2) {
                return vector1.dot(vector2);
            }

            static T lerp(const T& vector1, const T& vector2, const K& value) {
                return vector1.lerp(vector2, value);
            }

            virtual K dot(const T& vector);

            virtual K magnitude() const;
            virtual K squareMagnitude() const;

            virtual T normalized() const;
            virtual void normalize();

            virtual K operator [] (const short& i) const;
            virtual K& operator [] (const short& i);
            virtual K operator * (const T& vector) const;
            virtual T operator + (const T& vector) const;
            virtual T operator - (const T& vector) const;
            virtual bool operator == (const T& vector) const;

            virtual T operator - () const;
            virtual T operator * (const K& k) const;
            virtual T operator / (const K& k) const;

            virtual void operator += (const T& vector);
            virtual void operator -= (const T& vector);
            virtual void operator *= (const K& k);
            virtual void operator /= (const K& k);

            virtual K angleBetween (const T& vector) const;

            virtual T lerp(const T& vector, const K& t) const;

            virtual bool isZero() const;
            virtual bool areEquals(const T& vector) const;

            virtual void print() const;
    };
}

#endif