#ifndef TEST_HPP
#define TEST_HPP

#define DEBUG

#if defined(DEBUG)
class Test {
    public:
        virtual void test();
};
#endif

#endif