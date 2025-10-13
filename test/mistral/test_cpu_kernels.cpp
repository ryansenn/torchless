#include "setup/context.h"

int test_sum(){
    Tensor x({1.1f,2.2f,3.3f,4.4f,5.5f,6.6f}, {6});
    float expected = 23.1f;

    if (!equals(sum(x), expected)){
        std::cout << "sum mismatch: got " << sum(x)
                  << ", expected " << expected << std::endl;
        return 1;
    }

    return 0;
}

int test_mul(){
    Tensor x({-1.5f, 0.0f, 2.5f, 4.2f}, {4});
    float scalar = -3.0f;
    Tensor expected({4.5f, 0.0f, -7.5f, -12.6f}, {4});

    mul(x, x, scalar);

    if (!equals(x, expected)){
        std::cout << "tensor mul mismatch" << std::endl;
        return 1;
    }

    return 0;
}

int test_pow(){
    Tensor x({2.0f, 3.0f, 4.0f}, {3});
    float power = 2.0f;
    Tensor expected({4.0f, 9.0f, 16.0f}, {3});

    pow(x, x, power);

    if (!equals(x, expected)){
        std::cout << "tensor pow mismatch" << std::endl;
        return 1;
    }

    return 0;
}


int test_sqrt(){
    Tensor x({1.0f, 4.0f, 9.0f, 16.0f}, {4});
    Tensor expected({1.0f, 2.0f, 3.0f, 4.0f}, {4});

    sqrt(x, x);

    if (!equals(x, expected)){
        std::cout << "tensor sqrt mismatch" << std::endl;
        return 1;
    }

    return 0;
}

RegisterTest sum_reg("test sum", &test_sum);
RegisterTest mul_reg("test mul", &test_mul);
RegisterTest pow_reg("test pow", &test_pow);
RegisterTest sqrt_reg("test sqrt", &test_sqrt);