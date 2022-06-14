#ifndef TEST_HPP
#define TEST_HPP

#include <assert.h>

class Test 
{
public: 
    Test();

    static void Test(bool expression) {
        assert(expression);
    }
};

#endif