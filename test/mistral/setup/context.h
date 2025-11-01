#include <string>
#include <iostream>
#include "../../../src/loader/parameters.h"
#include "../../../src/common/kernels.h"
#include "../../../src/model/mistral/modules.h"

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
inline Arena arena(1024*1024); // 4 MB

bool equals(float x, float y);
bool equals(Tensor& x, Tensor& y);