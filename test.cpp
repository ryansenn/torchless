#include <math.h>
#include <iostream>

void matmul(float* xout, float* x, float* w, int n, int d);

int success = 1;

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
            std::cout << "matmul test failed " << std::endl;
            success = 0;
            return;
       }
    }
}

int main() {
    test_matmul();

    if (success) {
        std::cout << "all tests passed" << std::endl;
    }
    return 0;
}