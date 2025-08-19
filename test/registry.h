#include <string>

struct TestCase {
    std::string name;
    int (*func)();
};

inline std::vector<TestCase> tests;

struct RegisterTest {
    RegisterTest(std::string name, int (*func)()){
        tests.push_back({name, func});
    }
};