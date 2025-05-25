#include <math.h>


static void matmul(float* xout, float* x, float* w, int n, int d){
    // W (d,n) @ x (n,) = xout (d,)

    for (int i=0; i<d; i++){
        float res = 0;

        for (int j=0; j<n; j++){
            res += w[i*n + j] * x[j];
        }

        xout[i] = res;
    }
}