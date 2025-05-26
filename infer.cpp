#include <math.h>
#include <iostream>

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


void rmsnorm(float* o, // output
             float* x, // input
             float* g, // scale per element
             int    n, // size
             float eps) // epsilon 
{
    float mean_squared = 0;
    for (int i=0;i<n;i++){
        mean_squared += x[i]*x[i];
    }
    mean_squared = mean_squared/n;
    float m = sqrt(mean_squared + eps);

    for (int i=0; i<n; i++){
        o[i] = x[i] * g[i] / m;
    }

}

// layernorm, softmax, gelu, silu, clip, rope, attn, block, forward