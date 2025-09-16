#include <math.h>
#include <iostream>
#include "context.h"
#include "../model/tensor.h"

void matmul_impl(float* xout, float* w, float* x, int d, int n);
void rmsnorm(float* o, float* x, float* g, int n, float eps);
void layernorm(float* o, float* x, float* scale, float* shift, int n, float eps);
void softmax(float* o, float* x, int n);

const float tolerance = 1e-6f;

int test_matmul(){
    // W(d,n) @ x(n,) = xout

    int d = 3;
    int n = 4;

    float* w = new float[d*n];
    float* x = new float[n];
    float* xout = new float[d];
    float* expected = new float[d];

    for (int i=0; i<d*n; i++){
        w[i] = i+1;
    }

    for (int i=0; i<n; i++){
        x[i] = 1;
    }

    expected[0] = 10;
    expected[1] = 26;
    expected[2] = 42;

    matmul_impl(xout, w, x, d, n);

    for (int i=0; i<d; i++){
        if (xout[i] != expected[i]){
            std::cout << "matmul test failed" << std::endl;
            return 1;
        }
    }

    delete[] w; delete[] x; delete[] xout; delete[] expected;
    return 0;
}

int test_rmsnorm(){
    int n = 5;
    float o[n];
    float x[5] = {1,2,3,4,5};
    float g[5] = {2,3,1,3,2};
    float eps = 0;

    float res[5] = {0.6030227, 1.8090681, 0.9045340, 3.6181361, 3.0151134};

    rmsnorm(o, x, g, n, eps);

    for (int i=0;i<n;i++){
        if (std::fabs(res[i] - o[i]) > tolerance){
            std::cout << "rmsnorm test failed" << std::endl;
            return 1;
        }
    }

    return 0;
}

int test_layernorm() {
    int n = 5;
    float x[5] = {1, 2, 3, 4, 5};
    float scale[5] = {1, 1, 1, 1, 1};
    float shift[5] = {0, 0, 0, 0, 0};
    float eps = 0;

    float o[5];
    float res[5] = {-1.4142135, -0.7071068, 0, 0.7071068, 1.4142135};

    layernorm(o, x, scale, shift, n, eps);

    for (int i = 0; i < n; i++) {
        if (std::fabs(res[i] - o[i]) > tolerance) {
            std::cout << "layernorm test failed" << std::endl;
            return 1;
        }
    }

    return 0;
}

int test_softmax() {
    int n = 5;
    float x[5] = {2, 3, 1, 8, 4};

    float o[5];
    float res[5] = {0.0024102, 0.00655159, 0.00088666, 0.97234248, 0.01780907};

    softmax(o, x, n);

    for (int i = 0; i < n; i++) {
        if (std::fabs(res[i] - o[i]) > tolerance) {
            //std::cout << o[i] << " " << res[i] << std::endl;
            std::cout << "softmax test failed" << std::endl;
            return 1;
        }
    }

    return 0;
}

static RegisterTest reg_matmul("matmul", &test_matmul);
static RegisterTest reg_rmsnorm("rmsnorm", &test_rmsnorm);
static RegisterTest reg_layernorm("layernorm", &test_layernorm);
static RegisterTest reg_softmax("softmax", &test_softmax);

int test_tensor_index(){
    float data[4][4][2] = {
            { {0,1}, {2,3}, {4,5}, {6,7} },
            { {8,9}, {10,11}, {12,13}, {14,15} },
            { {16,17}, {18,19}, {20,21}, {22,23} },
            { {24,25}, {26,27}, {28,29}, {30,31} }
    };
    std::vector<int64_t> shape = {4, 4, 2};
    Tensor t("test", reinterpret_cast<float*>(data), shape);

    // Scalar access: t[1,2,0] == 12
    {
        Tensor v = t.at({1,2,0});
        if (!v.shape.empty() || *(v.data) != 12.f) {
            std::cout << "tensor_index scalar failed\n";
            return 1;
        }
    }

    // 1D slice: t[3,0] -> shape {2}, values {24,25}
    {
        Tensor v = t.at({3,0});
        if (v.shape.size() != 1 || v.shape[0] != 2) {
            std::cout << "tensor_index 1D shape failed\n";
            return 1;
        }
        if (v.data[0] != 24.f || v.data[1] != 25.f) {
            std::cout << "tensor_index 1D values failed\n";
            return 1;
        }
    }

    // 2D slice: t[2] -> shape {4,2}; check first and last elements (16, 23)
    {
        Tensor v = t.at({2});
        if (v.shape.size() != 2 || v.shape[0] != 4 || v.shape[1] != 2) {
            std::cout << "tensor_index 2D shape failed\n";
            return 1;
        }
        if (*(v.at({0,0}).data) != 16.f || *(v.at({3,1}).data) != 23.f) {
            std::cout << "tensor_index 2D values failed\n";
            return 1;
        }
    }

    return 0;
}

static RegisterTest reg_tensor_index("tensor_index", &test_tensor_index);

int test_tensor_reshape() {
    // Create a 2x3x4 tensor filled with values 0..23
    float data[24];
    for (int i = 0; i < 24; i++) data[i] = static_cast<float>(i);

    std::vector<int64_t> shape = {2, 3, 4};
    Tensor t("reshape_test", data, shape);

    // Reshape to 4x6
    Tensor r = t.reshape({4, 6});

    // Check shape
    if (r.shape.size() != 2 || r.shape[0] != 4 || r.shape[1] != 6) {
        std::cout << "tensor_reshape shape failed\n";
        return 1;
    }

    // Check that data is shared and values line up
    for (int i = 0; i < 24; i++) {
        if (r.data[i] != static_cast<float>(i)) {
            std::cout << "tensor_reshape data mismatch at " << i << "\n";
            return 1;
        }
    }

    // Now reshape back to original and check again
    Tensor r2 = r.reshape({2, 3, 4});
    if (r2.shape.size() != 3 || r2.shape[0] != 2 || r2.shape[1] != 3 || r2.shape[2] != 4) {
        std::cout << "tensor_reshape roundtrip shape failed\n";
        return 1;
    }
    if (r2.data[5] != 5.f || r2.data[23] != 23.f) {
        std::cout << "tensor_reshape roundtrip values failed\n";
        return 1;
    }

    return 0;
}

static RegisterTest reg_tensor_reshape("tensor_reshape", &test_tensor_reshape);