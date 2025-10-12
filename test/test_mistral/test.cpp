#include "context.h"
#include <iostream>

int main() {
    int total   = 0;
    int failed  = 0;

    for (const TestCase& t : tests) {
        total++;
        int result = t.func();
        if (result != 0) {
            failed++;
            std::cout << t.name << " has failed\n";
        }
    }

    int passed = total - failed;
    std::cout << "\nSummary: " << passed << " / " << total << " tests passed\n";

    return 0;
}