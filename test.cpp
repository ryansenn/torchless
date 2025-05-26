#include <math.h>
#include <iostream>

void matmul(float* xout, float* x, float* w, int n, int d);
void rmsnorm(float* o, float* x, float* g, int n, float eps);
void layernorm(float* o, float* x, float* scale, float* shift, int n, float eps);

int success = 1;
const float tolerance = 1e-6f;

void test_matmul(){
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

    matmul(xout, x, w, n, d);

    for (int i=0; i<d; i++){
       if (xout[i] != expected[i]){
            std::cout << "matmul test failed" << std::endl;
            success = 0;
            return;
       }
    }

    delete[] w; delete[] x; delete[] xout; delete[] expected;
}

void test_rmsnorm(){
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
            success = 0;
            return;
        }
    }
}

void test_layernorm() {
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
            success = 0;
            return;
        }
    }
}

int main() {
    test_matmul();
    test_rmsnorm();
    test_layernorm();

    if (success) {
        std::cout << "all tests passed" << std::endl;
    }
    return 0;
}