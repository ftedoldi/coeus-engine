#ifndef TESTMATRIX3_HPP
#define TESTMATRIX3_HPP

#include <Test.hpp>
#include <Matrix3.hpp>
#include <Vector3.hpp>

using namespace Athena;

class TestMatrix3 : public Test
{
public:

    void test()
    {
        //costructor test
        //...
        Matrix3 mTest((Scalar)2.0f, (Scalar)7.0f, (Scalar)8.0f, (Scalar)8.0f, (Scalar)5.0f, (Scalar)2.0f, (Scalar)4.0f, (Scalar)8.0f, (Scalar)9.0f);

        Matrix3 mTest1((Scalar)3.0f, (Scalar)2.0f, (Scalar)5.0f, (Scalar)4.0f, (Scalar)1.0f, (Scalar)6.0f, (Scalar)3.0f, (Scalar)7.0f, (Scalar)2.0f);

        Vector3 vTest((Scalar)1.0f, (Scalar)3.0f, (Scalar)2.0f);

        //***************INVERSE TESTS***************
        Matrix3 inverseResult((Scalar)-29.0f / 38.0f, (Scalar)-1.0f / 38.0f, (Scalar)13.0f / 19.0f,
                              (Scalar)32.0f / 19.0f, (Scalar)7.0f / 19.0f, (Scalar)-30.0f / 19.0f,
                              (Scalar)-22.0f / 19.0f, (Scalar)-6.0f / 19.0f, (Scalar)23.0f / 19.0f);
        Matrix3 inverse;

        inverse = Matrix3::inverse(mTest);
        assert(inverse == inverseResult);

        inverse = mTest.inverse();
        assert(inverse == inverseResult);

        //***************OPERATOR* TESTS***************

        //Matrix-Matrix product
        Matrix3 matProdResult((Scalar)58.0f, (Scalar)67.0f, (Scalar)68.0f,
                              (Scalar)50.0f, (Scalar)35.0f, (Scalar)74.0f,
                              (Scalar)71.0f, (Scalar)79.0f, (Scalar)86.0f);
        
        Matrix3 matProduct = mTest * mTest1;
        assert(matProduct == matProdResult);

        //Matrix-Vector product
        Vector3 vecProdResult((Scalar)39.0f, (Scalar)27.0f, (Scalar)46.0f);

        Vector3 vecProduct = mTest * vTest;
        assert(vecProduct == vecProdResult);

        //***************TRANSPOSE TESTS***************

        Matrix3 transposeResult((Scalar)2.0f, (Scalar)8.0f, (Scalar)4.0f,
                                (Scalar)7.0f, (Scalar)5.0f, (Scalar)8.0f,
                                (Scalar)8.0f, (Scalar)2.0f, (Scalar)9.0f);

        Matrix3 transpose;

        transpose = Matrix3::transpose(mTest);
        assert(transpose == transposeResult);

        transpose = mTest.transpose();
        assert(transpose == transposeResult);



    }
};

#endif