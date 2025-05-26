#include <math.h>

// W (d,n) @ x (n,) = xout (d,)
void matmul(float* xout, float* x, float* w, int n, int d){
    for (int i=0; i<d; i++){
        float res = 0;

        for (int j=0; j<n; j++){
            res += w[i*n + j] * x[j];
        }
        xout[i] = res;
    }
}


void rmsnorm(float* o, float* x, float* weight, int size, float eps){

}

// layernorm, softmax, gelu, silu, clip, rope, attn, block, forward