#include <math.h>
#include <iostream>
#include "context.h"
#include "../inference/math_ops.h"

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

    softmax(o, x, n, 1);

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

int test_rowvec_matmul(){
    // x(d,) * W(d,n) = xout(d,)

    int d = 3;
    int n = 4;

    float* w = new float[d*n];
    float* x = new float[d];
    float* xout = new float[n];
    float* expected = new float[n];

    for (int i=0; i<d*n; i++){
        w[i] = i+1;
    }

    for (int i=0; i<n; i++){
        x[i] = 1;
    }

    expected[0] = 15;
    expected[1] = 18;
    expected[2] = 21;

    rowvec_matmul_impl(xout, x, w, d, n);

    for (int i=0; i<d; i++){
        std::cout << xout[i] << " " << expected[i] << std::endl;
        if (xout[i] != expected[i]){
            return 1;
        }
    }

    delete[] w; delete[] x; delete[] xout; delete[] expected;
    return 0;
}

static RegisterTest reg_rowvec_matmul("rowvec_matmul", &test_rowvec_matmul);
