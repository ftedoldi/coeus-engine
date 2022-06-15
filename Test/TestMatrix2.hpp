#ifndef TESTMATRIX2_HPP
#define TESTMATRIX2_HPP

#include <Test.hpp>
#include <Matrix2.hpp>
#include <Vector2.hpp>

using namespace Athena;

class TestMatrix2 : public Test
{
public:

    void test()
    {
        //costructor test
        Matrix2 mTest((Scalar)1.0f, (Scalar)2.0f, (Scalar)3.0f, (Scalar)4.0f);

        assert(mTest.data[0] == (Scalar)1.0f);
        assert(mTest.data[1] == (Scalar)2.0f);
        assert(mTest.data[2] == (Scalar)3.0f);
        assert(mTest.data[3] == (Scalar)4.0f);

        Matrix2 mTest1((Scalar)2.0f, (Scalar)3.0f, (Scalar)4.0f, (Scalar)5.0f);
        Vector2 vTest((Scalar)2.0f, (Scalar)3.0f);

        //***************INVERSE TESTS***************
        Matrix2 inverseResult((Scalar) -5.0f / 2.0f, (Scalar)3.0f / 2.0f, (Scalar)2.0f, (Scalar)-1.0f);
        Matrix2 inverse;

        inverse = Matrix2::inverse(mTest1);
        assert(inverse == inverseResult);

        inverse = mTest1.inverse();
        assert(inverse == inverseResult);

        //***************OPERATOR TESTS***************
        //Matrix Matrix product
        Matrix2 matProduct = mTest * mTest1;
        Matrix2 matProdResult((Scalar)10.0f, (Scalar)13.0f, (Scalar)22.0f, (Scalar)29.0f);
        assert(matProduct == matProdResult);

        //Matrix Vector product
        Vector2 vecProduct = mTest * vTest;
        Vector2 vecProdResult((Scalar)8.0f, (Scalar)18.0f);
        assert(vecProduct == vecProdResult);

        //***************TRANSPOSE TESTS***************
        
        Matrix2 transposeResult((Scalar)1.0f, (Scalar)3.0f, (Scalar)2.0f, (Scalar)4.0f);
        Matrix2 transpose;

        transpose = Matrix2::transpose(mTest);
        assert(transpose == transposeResult);

        transpose = mTest.transpose();
        assert(transpose == transposeResult);

    }
};

#endif