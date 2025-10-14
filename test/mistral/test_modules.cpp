#include "setup/context.h"

int test_rmsnorm() {
    Tensor x({2.4f, 7.8f, 1.1f, 5.3f, 9.0f, 4.7f, 6.2f, 3.8f, 8.5f, 0.6f}, {10});
    Tensor g({0.8f, 1.2f, 1.0f, 0.9f, 1.1f, 1.3f, 0.7f, 1.0f, 1.2f, 0.95f}, {10});

    RMSNorm rms(g);
    Tensor out = rms.forward(x);

    Tensor expected({0.3371f, 1.6432f, 0.1931f, 0.8374f, 1.7380f, 1.0726f, 0.7619f, 0.6671f, 1.7906f, 0.1001f}, {10});

    if (!equals(out, expected)) {
        std::cout << "RMSNorm mismatch" << std::endl;
        return 1;
    }

    return 0;
}

RegisterTest rmsnorm_reg("test rmsnorm", &test_rmsnorm);