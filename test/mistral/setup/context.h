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

std::shared_ptr<Parameters> get_params();
inline InferenceState infer(get_params()->config);
inline Arena arena(1024*1024); // 4 MB

bool equals(float x, float y);
bool equals(const Tensor& x, const Tensor& y);