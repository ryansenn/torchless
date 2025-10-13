#include <string>
#include <iostream>
#include "../../../src/loader/parameters.h"
#include "../../../src/common/kernels.h"

struct TestCase {
    std::string name;
    int (*func)();
};

extern std::vector<TestCase> tests;

struct RegisterTest {
    RegisterTest(std::string name, int (*func)()){
        tests.push_back({name, func});
    }
};

Model& get_model();

bool equals(float x, float y);